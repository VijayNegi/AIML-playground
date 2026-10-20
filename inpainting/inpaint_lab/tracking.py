"""Aim experiment tracking: mirror runs and sync human reviews."""

import json
import sys

from inpaint_lab.config import AIM_REPO
from inpaint_lab.runs import RATINGS


def track_aim_run(args, run_dir, perf, result, comparison):
    """Mirror a completed local run into Aim without risking saved results."""
    run = None
    aim_hash = None
    try:
        from aim import Run, Image as AimImage

        run = Run(repo=str(AIM_REPO), experiment="collage",
                  system_tracking_interval=None, log_system_params=False,
                  capture_terminal_logs=False)
        run.name = run_dir.name
        run["params"] = vars(args)
        run["run_id"] = run_dir.name
        run["run_dir"] = str(run_dir.resolve())
        run["metadata"] = {name: value for name, value in perf.items()
                           if not isinstance(value, (int, float))}
        for name, value in perf.items():
            if isinstance(value, (int, float)):
                run.track(value, name=name, step=0)
        run.track(AimImage(result), name="output", step=0)
        run.track(AimImage(comparison), name="comparison", step=0)
        print(f"Aim run: {run.hash}")
        aim_hash = str(run.hash)
    except Exception as error:
        print(f"Warning: Aim tracking failed: {error}. Local results are saved in {run_dir}", file=sys.stderr)
    finally:
        if run is not None:
            try:
                run.close()
            except Exception as error:
                aim_hash = None
                print(f"Warning: Aim close failed: {error}", file=sys.stderr)
    return aim_hash


def sync_review(run_dir, review):
    """Update one existing Aim run; never create a replacement or duplicate run."""
    repo = None
    run = None
    try:
        from aim import Repo, Run, Text

        if any(type(review.get(name)) is not int or not 1 <= review[name] <= 5 for name in RATINGS):
            raise ValueError("Review ratings must be whole numbers from 1 to 5")
        repo = Repo(str(AIM_REPO))
        matches = []
        already_synced = False
        for candidate in repo.iter_runs():
            try:
                if candidate.get("run_dir") == str(run_dir.resolve()):
                    matches.append(candidate.hash)
                    already_synced = review.get("reviewed_at") is not None and candidate.get("review_synced_at") == review["reviewed_at"]
            finally:
                try:
                    candidate.close()
                except AttributeError as error:
                    # shortcut: Aim 3 read-only close lacks sequence_infos; remove when Aim fixes its cleanup.
                    if "sequence_infos" not in str(error):
                        raise
        if len(matches) != 1:
            raise ValueError(f"Expected one matching Aim run, found {len(matches)}")
        if already_synced:
            return
        run = Run(matches[0], repo=repo, system_tracking_interval=None,
                  log_system_params=False, capture_terminal_logs=False)
        for name in RATINGS:
            run.track(review[name], name=f"review.{name}", step=0)
        run.track(Text(review.get("notes", "")), name="review.notes", step=0)
        run["review_synced_at"] = review.get("reviewed_at")
        if run.get("review_synced_at") != review.get("reviewed_at"):
            raise RuntimeError("Aim did not save the review sync timestamp")
        if run.get("review") is not None:
            del run["review"]
        print(f"Synced review to Aim: {matches[0]}")
    except Exception as error:
        print(f"Warning: Aim review sync failed for {run_dir}: {error}. Local review is preserved; retry by running this script again.")
    finally:
        for resource in (run, repo):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    print(f"Warning: Aim close failed: {error}")


def sync_saved_reviews(runs_dir):
    for review_path in sorted(runs_dir.glob("*/review.json")):
        try:
            sync_review(review_path.parent, json.loads(review_path.read_text()))
        except (OSError, ValueError) as error:
            print(f"Cannot load {review_path}: {error}")
