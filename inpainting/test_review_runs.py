"""Visual review checks without opening an image viewer."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import threading
import unittest
from urllib.parse import urlencode
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

from PIL import Image

from review_runs import pending_runs, review_server, sync_saved_reviews, sync_review


class ReviewTests(unittest.TestCase):
    def test_sync_updates_existing_run_and_skips_identical_review(self):
        directory = Path("runs/example")
        review = {"seam_quality": 4, "scene_coherence": 3, "preservation": 5, "notes": "Good", "reviewed_at": "2026-10-12T00:00:00+00:00"}
        candidate = MagicMock()
        candidate.hash = "existing-hash"
        candidate.close.side_effect = AttributeError("'RunTracker' object has no attribute 'sequence_infos'")
        candidate.get.side_effect = lambda key: str(directory.resolve()) if key == "run_dir" else None
        repo = MagicMock()
        repo.iter_runs.return_value = [candidate]
        run = MagicMock()
        run.get.side_effect = lambda key: review["reviewed_at"] if key == "review_synced_at" else review
        aim = types.ModuleType("aim")
        aim.Repo = MagicMock(return_value=repo)
        aim.Run = MagicMock(return_value=run)
        aim.Text = MagicMock(side_effect=lambda text: text)
        with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()):
            sync_review(directory, review)
        self.assertEqual(aim.Run.call_args.args, ("existing-hash",))
        run.__setitem__.assert_called_once_with("review_synced_at", review["reviewed_at"])
        run.track.assert_any_call(4, name="review.seam_quality", step=0)
        run.track.assert_any_call("Good", name="review.notes", step=0)
        run.__delitem__.assert_called_once_with("review")
        run.close.assert_called_once()
        candidate.close.assert_called_once()
        repo.close.assert_called_once()
        aim.Run.reset_mock()
        candidate.get.side_effect = lambda key: str(directory.resolve()) if key == "run_dir" else review["reviewed_at"]
        candidate.__getitem__.return_value = {"review_synced_at": review["reviewed_at"]}
        with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()):
            sync_review(directory, review)
        aim.Run.assert_not_called()

    def test_missing_or_ambiguous_aim_run_never_creates_run(self):
        for count in (0, 2):
            repo = MagicMock()
            candidates = [MagicMock() for _ in range(count)]
            for candidate in candidates:
                candidate.get.return_value = str(Path("runs/example").resolve())
            repo.iter_runs.return_value = candidates
            aim = types.ModuleType("aim")
            aim.Repo = MagicMock(return_value=repo)
            aim.Run = MagicMock()
            aim.Text = MagicMock()
            with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()) as output:
                sync_review(Path("runs/example"), {"seam_quality": 4, "scene_coherence": 3, "preservation": 5})
            aim.Run.assert_not_called()
            self.assertIn("Local review is preserved", output.getvalue())

    def make_run(self, root, name):
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "params.json").write_text(json.dumps({"output": f"runs/{name}/output.png", "seed": 42}))
        for filename in ("output.png", "output_comparison.png"):
            Image.new("RGB", (8, 8)).save(run_dir / filename)
        return run_dir

    def test_only_completed_unreviewed_runs_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = self.make_run(root, "pending")
            reviewed = self.make_run(root, "reviewed")
            (reviewed / "review.json").write_text("{}")
            incomplete = self.make_run(root, "incomplete")
            (incomplete / "output_comparison.png").unlink()
            self.assertEqual([entry[0] for entry in pending_runs(root)], [pending])

    def test_browser_review_validation_save_and_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self.make_run(root, "example")
            with review_server(root) as server, patch("review_runs.sync_review") as sync:
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                def request(method, path, fields=None):
                    connection.request(method, path, body=urlencode(fields) if fields else None,
                                       headers={"Content-Type": "application/x-www-form-urlencoded"})
                    response = connection.getresponse()
                    return response.status, response.read()
                try:
                    status, page = request("GET", "/")
                    self.assertEqual(status, 200)
                    self.assertIn(b'Save review and sync to Aim', page)
                    self.assertEqual(request("GET", "/image?run=example")[0], 200)
                    self.assertEqual(request("GET", "/image?run=../secret")[0], 404)
                    fields = {"run": "example", "token": "wrong", "seam_quality": "4",
                              "scene_coherence": "3", "preservation": "5", "notes": "Good seam"}
                    self.assertEqual(request("POST", "/review", fields)[0], 403)
                    fields["token"] = server.token
                    fields["seam_quality"] = "6"
                    self.assertEqual(request("POST", "/review", fields)[0], 400)
                    self.assertFalse((run_dir / "review.json").exists())
                    fields["seam_quality"] = "4"
                    self.assertEqual(request("POST", "/review", fields)[0], 303)
                    review = json.loads((run_dir / "review.json").read_text())
                    self.assertEqual(review["seam_quality"], 4)
                    self.assertEqual(review["notes"], "Good seam")
                    sync.assert_called_once_with(run_dir, review)
                    self.assertEqual(request("POST", "/review", fields)[0], 409)
                    self.assertIn(b'All available runs reviewed', request("GET", "/")[1])
                finally:
                    connection.close()
                    server.shutdown()
                    thread.join()
            with patch("review_runs.sync_review") as retry:
                sync_saved_reviews(root)
            retry.assert_called_once_with(run_dir, review)


if __name__ == "__main__":
    unittest.main()
