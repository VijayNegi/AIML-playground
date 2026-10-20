"""Review completed runs in a local browser and sync saved reviews to Aim."""

import argparse
from datetime import datetime
import json
import io
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import math
from pathlib import Path
import secrets
from urllib.parse import parse_qs, quote, urlsplit
import webbrowser

from PIL import Image


RATINGS = {
    "seam_quality": "Seam quality (1=obvious seam, 5=invisible seam)",
    "scene_coherence": "Scene coherence (1=implausible/artifacts, 5=believable)",
    "preservation": "Preservation (1=unwanted changes, 5=original content preserved)",
}


def pending_runs(runs_dir):
    """Find recorded runs with saved outputs and no existing visual review."""
    for params_path in sorted(runs_dir.glob("*/params.json")):
        run_dir = params_path.parent
        if (run_dir / "review.json").exists():
            continue
        try:
            params = json.loads(params_path.read_text())
            output = run_dir / Path(params["output"]).name
            comparison = output.with_name(f"{output.stem}_comparison.png")
            if output.is_file() and comparison.is_file():
                yield run_dir, params, comparison
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Skipping {run_dir}: {error}")


def sync_review(run_dir, review):
    """Update one existing Aim run; never create a replacement or duplicate run."""
    repo = None
    run = None
    try:
        from aim import Repo, Run, Text

        if any(type(review.get(name)) is not int or not 1 <= review[name] <= 5 for name in RATINGS):
            raise ValueError("Review ratings must be whole numbers from 1 to 5")
        repo = Repo(str(Path(__file__).resolve().parent))
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


def metric_regions(name, metrics):
    """Return measured rectangles in output pixels, not a defect heatmap."""
    size = metrics.get("canvas_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        return []
    width, height = size
    def valid_box(value):
        if (isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value)
                and 0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height):
            return value
        return None
    def box(key):
        return valid_box(metrics.get(key))
    if name == "quality_maniqa":
        if "quality_maniqa_tiles" in metrics:
            return [region for tile in metrics["quality_maniqa_tiles"]
                    if (region := valid_box(tile.get("bbox"))) is not None]
        region = box("quality_maniqa_crop_bbox")
        return [region] if region else []
    if name == "quality_maniqa_min":
        region = box("quality_maniqa_min_bbox")
        return [region] if region else []
    if name.startswith("quality_maniqa_tile_"):
        index = name.removeprefix("quality_maniqa_tile_")
        tiles = metrics.get("quality_maniqa_tiles", [])
        if index.isdigit() and int(index) < len(tiles):
            region = valid_box(tiles[int(index)].get("bbox"))
            return [region] if region else []
        return []
    if name.startswith("quality_preservation_"):
        region = box("quality_editable_bbox")
        if not region:
            return []
        left, _, right, _ = region
        return [r for r in ([0, 0, left, height], [right, 0, width, height]) if r[0] < r[2]]
    for group, key in (("gap", "quality_gap_bbox"), ("mask", "quality_effective_mask_bbox"),
                       ("editable", "quality_editable_bbox"), ("patch", "crop_bbox")):
        region = box(key)
        if not region:
            continue
        for edge in ("left", "right", "top", "bottom"):
            prefix = f"quality_{group}_{edge}_"
            if not name.startswith(prefix):
                continue
            vertical = edge in ("left", "right")
            position = region[{"left": 0, "top": 1, "right": 2, "bottom": 3}[edge]]
            extent = width if vertical else height
            if not 0 < position < extent:
                return []
            band_width = metrics.get("quality_band_width")
            if type(band_width) is not int or band_width <= 0:
                return []
            band = min(band_width, position, extent - position)
            def strip(start, end):
                return [start, region[1], end, region[3]] if vertical else [region[0], start, region[2], end]
            suffix = name[len(prefix):]
            if suffix == "delta_e76":
                return [strip(position - band, position), strip(position, position + band)]
            boundary = strip(position - 1, position + 1)
            neighbors = [strip(max(0, position - band - 1), position),
                         strip(position, min(extent, position + band + 1))]
            if suffix == "nearby_gradient":
                return neighbors
            if suffix == "gradient_excess":
                return [boundary] + neighbors
            if suffix == "gradient":
                return [boundary]
    return []


def automatic_scores_html(run_dir):
    """Display recorded scores without recalculating or inventing missing ratings."""
    load_error = ""
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be an object")
    except (OSError, ValueError):
        metrics = {}
        load_error = "Automatic scores unavailable: metrics.json is missing or invalid."
    def number(key, percent=False):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            return "Not available"
        return f"{value * 100:.2f}%" if percent else f"{value:.3f}"

    def highlight(name, label="Highlight measured area"):
        regions = metric_regions(name, metrics)
        if not regions:
            return '<small>Area unavailable: this run has no recorded geometry for this metric.</small>'
        return (f'<button type="button" class="metric-highlight" aria-pressed="false" '
                f'data-metric="{escape(name, quote=True)}" data-regions="{escape(json.dumps(regions), quote=True)}">{label}</button>')

    maniqa = number("quality_maniqa")
    if maniqa == "Not available":
        if metrics.get("quality_maniqa_error"):
            maniqa_help = "Scoring failed for this run. See the assessment warning below."
            if "pyiqa" in str(metrics["quality_maniqa_error"]) and "No module named" in str(metrics["quality_maniqa_error"]):
                maniqa_help += " Activate your virtual environment, install requirements.txt, then start a new collage run."
        elif metrics.get("quality_error"):
            maniqa_help = "Quality assessment failed before a MANIQA score was recorded."
        else:
            maniqa_help = "No MANIQA score was recorded for this run. Older runs are not scored retroactively."
    else:
        maniqa_help = "Higher suggests better visual quality. Usually around 0–1; not a percentage or a human 1–5 rating. Compare the same input pair."
    def cell(name, label):
        value = number(name)
        regions = metric_regions(name, metrics)
        if value == "Not available":
            return '<span title="Not recorded or not measurable for this run">Unavailable</span>'
        if not regions:
            return f'<span title="{escape(label, quote=True)}; {escape(name, quote=True)}; measured area unavailable">{value}</span>'
        return (f'<button type="button" class="metric-highlight" aria-pressed="false" '
                f'aria-label="{escape(label, quote=True)}: {value}" title="{escape(name, quote=True)}" '
                f'data-metric="{escape(name, quote=True)}" data-regions="{escape(json.dumps(regions), quote=True)}">{value}</button>')

    def boundary_table(boundaries):
        columns = (("delta_e76", "Color mismatch"), ("gradient_excess", "Extra edge strength"),
                   ("gradient", "Edge strength"), ("nearby_gradient", "Nearby edges"))
        rows = []
        for boundary, label in boundaries:
            if not any(f"quality_{boundary}_{suffix}" in metrics for suffix, _ in columns):
                continue
            cells = "".join(f'<td>{cell(f"quality_{boundary}_{suffix}", f"{label}: {title}")}</td>' for suffix, title in columns)
            rows.append(f'<tr><th scope="row">{label}</th>{cells}</tr>')
        if not rows:
            return '<p>No boundary scores recorded for this run.</p>'
        return ('<div class="metric-table"><table><thead><tr><th scope="col">Measured edge</th>' +
                "".join(f'<th scope="col">{title}</th>' for _, title in columns) +
                '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')

    tile_html = ""
    if metrics.get("quality_maniqa_tiles"):
        tile_buttons = " ".join(highlight(f"quality_maniqa_tile_{index}", f"Tile {index + 1}: {tile['score']:.3f}")
                                for index, tile in enumerate(metrics["quality_maniqa_tiles"]))
        tile_html = (f'<p><strong>Lowest tile: {number("quality_maniqa_min")}</strong> '
                     f'{highlight("quality_maniqa_min", "Show lowest tile")}</p>'
                     f'<p>{len(metrics["quality_maniqa_tiles"])} overlapping 224×224 tiles, top to bottom. '
                     'Summary is the median; the lowest tile is a possible weak spot, not an automatic failure.</p>'
                     f'<p class="tile-scores">{tile_buttons}</p>')
    maniqa_scope = ("Native-resolution tiles follow the gap down its full height. No overview resizing is used."
                    if metrics.get("quality_maniqa_tiles") else
                    "This run has no tiled scores. Any recorded MANIQA score uses the older context-overview assessment.")
    cards = f'''<section class="quality-panel" aria-labelledby="automatic-heading"><h2 id="automatic-heading">Automatic assessment</h2>
<p>Priority: gap quality, then seams, then patch checks. Select a value to highlight its area. No calibrated pass/fail thresholds.</p>
<section class="quality-group gap-quality"><h3>1. Gap quality · most important</h3>
<p><strong title="quality_maniqa">Visual quality · MANIQA: {maniqa}</strong> {highlight("quality_maniqa", "Show area")}</p>
<p>{escape(maniqa_help)} {maniqa_scope}</p>{tile_html}</section>
<section class="quality-group"><h3>2. Seams · does the join look continuous?</h3>
{boundary_table((("mask_left", "Left edge of model mask"), ("mask_right", "Right edge of model mask"), ("gap_left", "Left edge of gap"), ("gap_right", "Right edge of gap"), ("editable_left", "Left outer feather edge"), ("editable_right", "Right outer feather edge")))}
<p>The gap, outer feather boundary, and model mask are different edges. SDXL thresholds feather gray values at 50%, so its model-mask edge falls inside the feather. Older runs may only have nominal outer feather bounds.</p>
<p>Color mismatch: Lab distance, 0 upward; lower means closer colors. Extra edge strength: −1 to 1; positive means sharper than nearby edges. Edge strength and nearby edges: 0–1. Real shadows or object edges can legitimately score high.</p></section>
<section class="quality-group"><h3>3. Patch · unwanted changes and pasted-crop edges</h3>
<p><strong>Original pixels changed: {number("quality_preservation_changed_fraction", percent=True)}</strong> {highlight("quality_preservation_changed_fraction", "Show area")}
<br><strong>Size of unwanted changes: {number("quality_preservation_mae")}</strong> {highlight("quality_preservation_mae", "Show area")}</p>
<p>Outside the gap and feather zone. Lower is better: 0% means unchanged; average change uses 0–255 color levels. Even invisible pixel changes count.</p>
{boundary_table((("patch_left", "Left edge of pasted crop"), ("patch_right", "Right edge of pasted crop"), ("patch_top", "Top edge of pasted crop"), ("patch_bottom", "Bottom edge of pasted crop")))}
<p>Same seam scales above; checks where the full model-generated crop was pasted back. Canvas-border edges have no outside neighbor and are omitted.</p></section>'''
    if load_error:
        cards += f"<p role='status'>{escape(load_error)}</p>"
    errors = " ".join(str(metrics[key]) for key in ("quality_error", "quality_maniqa_error") if key in metrics)
    if errors:
        cards += f"<p role='status'>Assessment warning: {escape(errors)}</p>"
    return cards + "</section>"


def surrounding_metrics_html(run_dir):
    """Place range rulers and tile values around the image using saved geometry."""
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        width, height = metrics["canvas_size"]
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("Invalid canvas dimensions")
    except (OSError, ValueError, KeyError, TypeError):
        return "", "", '<p>No recorded spatial metrics.</p>'

    def control(name, label):
        regions = metric_regions(name, metrics)
        value = metrics.get(name)
        if not regions or type(value) not in (int, float) or not math.isfinite(value):
            return ""
        return (f'<button type="button" class="metric-highlight" aria-pressed="false" '
                f'data-metric="{escape(name, quote=True)}" data-regions="{escape(json.dumps(regions), quote=True)}" '
                f'title="{escape(name, quote=True)}">{label} {value:.3g}</button>')

    def ruler(group, key, label):
        bbox = metrics.get(key)
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox)
                and 0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
            return ""
        values = " ".join(control(f"quality_{group}_{edge}_{metric}", f"{side} {short}")
                          for edge, side in (("left", "L"), ("right", "R"))
                          for metric, short in (("delta_e76", "color"), ("gradient_excess", "extra edge")))
        return (f'<div class="range-row"><div><strong>{label}</strong> <small>x {bbox[0]}–{bbox[2]}</small> {values}</div>'
                f'<div class="range-track" aria-label="{label} horizontal range"><span style="left:{bbox[0] / width * 100:.6f}%;'
                f'width:{(bbox[2] - bbox[0]) / width * 100:.6f}%"></span></div></div>')

    top = ruler("gap", "quality_gap_bbox", "Gap") + ruler("mask", "quality_effective_mask_bbox", "Model mask")
    bottom = ruler("editable", "quality_editable_bbox", "Outer feather") + ruler("patch", "crop_bbox", "Pasted crop")
    tiles = metrics.get("quality_maniqa_tiles", [])
    lines, buttons = [], []
    for index, tile in enumerate(tiles):
        regions = metric_regions(f"quality_maniqa_tile_{index}", metrics)
        if not regions or type(tile.get("score")) not in (int, float) or not math.isfinite(tile["score"]):
            continue
        center = (regions[0][1] + regions[0][3]) / 2
        label_y = (index + 0.5) / len(tiles)
        lines.append(f'<line x1="0" y1="{center}" x2="22" y2="{label_y * height}"/>')
        buttons.append(f'<button type="button" class="metric-highlight tile-value" aria-pressed="false" '
                       f'data-metric="quality_maniqa_tile_{index}" data-regions="{escape(json.dumps(regions), quote=True)}" '
                       f'style="top:{label_y * 100:.6f}%" title="Tile {index + 1}; y {regions[0][1]}–{regions[0][3]}; higher is better">'
                       f'{index + 1}: {tile["score"]:.3f}</button>')
    side = (f'<svg class="tile-connectors" viewBox="0 0 100 {height}" preserveAspectRatio="none" aria-hidden="true">'
            + "".join(lines) + '</svg>' + "".join(buttons)) if buttons else '<p>No tiled MANIQA scores.</p>'
    return top, bottom, side


class ReviewHandler(BaseHTTPRequestHandler):
    def valid_host(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(403, "Use the printed loopback URL")
            return False
        return True

    def respond(self, body, content_type="text/html; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.valid_host():
            return
        address = urlsplit(self.path)
        if address.path == "/batches" or address.path == "/batch.js" or address.path.startswith("/api/batches"):
            from batch_ui import get_route
            if get_route(self, address):
                return
        if address.path == "/review.js":
            self.respond(Path(__file__).with_name("review.js").read_bytes(), "text/javascript; charset=utf-8")
            return
        runs = list(pending_runs(self.server.runs_dir))
        name = parse_qs(address.query).get("run", [""])[0]
        selected = next((entry for entry in runs if entry[0].name == name), None)
        if address.path == "/image":
            if selected is None:
                self.send_error(404)
                return
            view = parse_qs(address.query).get("view", ["output"])[0]
            if view not in ("input", "output"):
                self.send_error(400, "Unknown image view")
                return
            try:
                image_path = selected[2] if view == "input" else selected[0] / Path(selected[1]["output"]).name
                with Image.open(image_path) as image:
                    if view == "input":
                        image = image.crop((0, 0, image.width, image.height // 2))
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                self.respond(buffer.getvalue(), "image/png")
            except OSError as error:
                self.send_error(404, str(error))
            return
        if address.path != "/":
            self.send_error(404)
            return
        if name and selected is None:
            self.send_error(404, "Run is missing or already reviewed")
            return
        selected = selected or (runs[0] if runs else None)
        content = "<h1>All available runs reviewed</h1>"
        if selected:
            run_dir, params, _ = selected
            options = "".join(f'<option value="{escape(entry[0].name, quote=True)}"'
                              f'{" selected" if entry == selected else ""}>{escape(entry[0].name)}</option>'
                              for entry in runs)
            fields = "".join(f'<fieldset><legend>{escape(label)}</legend>' + "".join(
                f'<label><input type="radio" name="{key}" value="{value}" required> {value}</label> '
                for value in range(1, 6)) + '</fieldset>' for key, label in RATINGS.items())
            index = runs.index(selected)
            next_run = runs[(index + 1) % len(runs)][0].name
            top_ranges, bottom_ranges, tile_values = surrounding_metrics_html(run_dir)
            content = f'''<h1>Visual review</h1><p>{len(runs)} unreviewed runs</p>
<form method="get"><label>Choose run <select name="run">{options}</select></label><button>Open</button></form>
<h2>{escape(run_dir.name)}</h2><details><summary>All settings</summary><pre>{escape(json.dumps(params, indent=2))}</pre></details>
<p>Showing: <strong id="image-view-label">Output</strong>. Input is the assembled canvas before inpainting. Click image for full resolution.</p>
<button type="button" id="flip-image">Show input</button>
<p class="range-help">Range markers use image coordinates. L/R = left/right. Color = Lab difference; extra edge = sharper than nearby edges. Hover/focus or select values to inspect. MANIQA: higher is better.</p>
<div class="metric-workspace"><div class="top-ranges">{top_ranges}</div><strong class="tile-heading">MANIQA tiles</strong>
<div class="comparison-frame"><a id="full-image-link" href="/image?run={quote(run_dir.name)}&amp;view=output" target="_blank" rel="noopener"><img id="comparison-image" alt="Inpainted output" src="/image?run={quote(run_dir.name)}&amp;view=output" data-output-src="/image?run={quote(run_dir.name)}&amp;view=output" data-input-src="/image?run={quote(run_dir.name)}&amp;view=input"></a><svg id="metric-overlay" aria-hidden="true"></svg></div>
<aside class="tile-rail" aria-label="MANIQA tile scores aligned to image height">{tile_values}</aside><div class="bottom-ranges">{bottom_ranges}</div></div>
<p id="highlight-status" role="status">Hover or focus a metric button to preview its measured area. Click to keep it selected.</p>
<button type="button" id="clear-highlight">Clear highlight</button>
{automatic_scores_html(run_dir)}
<form method="post" action="/review"><input type="hidden" name="token" value="{self.server.token}">
<input type="hidden" name="run" value="{escape(run_dir.name, quote=True)}">{fields}
<p><label>Notes<br><textarea name="notes" rows="3" maxlength="10000"></textarea></label></p>
<button>Save review and sync to Aim</button> <a href="/?run={quote(next_run)}">Skip / next</a></form>'''
        message = parse_qs(address.query).get("message", [""])[0]
        page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Inpainting review</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}}img{{width:100%;height:auto}}fieldset{{margin:16px 0}}button,select,textarea{{font:inherit}}textarea{{width:100%;box-sizing:border-box}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}label{{margin-right:12px}}table{{width:100%;border-collapse:collapse;overflow-wrap:anywhere}}th,td{{text-align:left;padding:8px;border:1px solid #ccc}}details{{margin:16px 0}}</style>
<style>.quality-panel{{font-size:14px;border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0}}.quality-panel h2{{font-size:18px;margin:0 0 8px}}.quality-group{{border-top:1px solid #ddd;padding:8px 0}}.quality-group h3{{font-size:15px;margin:4px 0 8px}}.quality-group p{{margin:6px 0;line-height:1.4}}.gap-quality{{border-left:4px solid #1468ac;padding-left:10px}}.metric-table{{overflow-x:auto}}.quality-panel th,.quality-panel td{{padding:4px 6px}}.quality-panel .metric-highlight{{margin:0;font:inherit}}small{{font-weight:normal;overflow-wrap:anywhere}}</style>
<style>.comparison-frame{{position:relative}}.comparison-frame img{{display:block}}#metric-overlay{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.metric-highlight[aria-pressed="true"]{{outline:3px solid #1468ac}}.metric-highlight{{margin-top:8px}}</style>
<style>.metric-workspace{{display:grid;grid-template-columns:minmax(0,1fr) 112px;grid-template-rows:auto auto auto;gap:4px 8px;max-width:calc(55vh * var(--image-aspect,1.6) + 120px)}}.top-ranges{{grid-area:1/1}}.tile-heading{{grid-area:1/2;align-self:end;font-size:13px}}.comparison-frame{{grid-area:2/1}}.tile-rail{{grid-area:2/2;position:relative;font-size:12px}}.bottom-ranges{{grid-area:3/1}}.range-row{{font-size:12px;margin:3px 0}}.range-row button{{font:inherit;margin:0 2px;padding:1px 4px}}.range-track{{height:12px;position:relative;background:#f4f4f4}}.range-track span{{position:absolute;top:2px;height:7px;border:1px solid #1468ac;border-left-width:3px;border-right-width:3px;box-sizing:border-box;background:#1473e61a}}.tile-connectors{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.tile-connectors line{{stroke:#888;stroke-width:1;vector-effect:non-scaling-stroke}}.tile-value{{position:absolute;left:24%;transform:translateY(-50%);font:inherit;margin:0;padding:2px 4px;white-space:nowrap}}.range-help{{font-size:12px}}@media(max-width:600px){{.metric-workspace{{grid-template-columns:minmax(0,1fr) 80px}}.tile-value{{font-size:10px;padding:1px 2px}}}}</style>
</head><body><nav><a href="/">Human review</a> · <a href="/batches">Batch builder &amp; history</a></nav><p role="status">{escape(message)}</p>{content}<script src="/review.js" defer></script></body></html>'''
        self.respond(page.encode())

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path.startswith("/api/batches"):
            from batch_ui import post_route
            if post_route(self):
                return
        if self.path != "/review":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size")
            fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            if not secrets.compare_digest(fields.get("token", [""])[0], self.server.token):
                self.send_error(403, "Invalid form token")
                return
            name = fields.get("run", [""])[0]
            selected = next((entry for entry in pending_runs(self.server.runs_dir) if entry[0].name == name), None)
            if selected is None:
                self.send_error(409, "Run is missing or already reviewed")
                return
            review = {}
            for key in RATINGS:
                value = fields.get(key, [""])[0]
                if value not in ("1", "2", "3", "4", "5"):
                    raise ValueError("Each rating must be a whole number from 1 to 5")
                review[key] = int(value)
            review["notes"] = fields.get("notes", [""])[0].strip()
            if len(review["notes"]) > 10000:
                raise ValueError("Notes exceed 10000 characters")
            review["reviewed_at"] = datetime.now().astimezone().isoformat()
            run_dir = selected[0]
            with (run_dir / "review.json").open("x") as file:
                file.write(json.dumps(review, indent=2) + "\n")
        except (ValueError, OSError) as error:
            self.send_error(400, str(error))
            return
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()) as messages:
            sync_review(run_dir, review)
        message = f"Review saved for {name}. {messages.getvalue().strip()}"
        self.send_response(303)
        self.send_header("Location", "/?message=" + quote(message))
        self.end_headers()


class ReviewServer(HTTPServer):
    def server_close(self):
        batch_manager = getattr(self, "batch_manager", None)
        if batch_manager is not None:
            batch_manager.close()
        super().server_close()


def review_server(runs_dir, port=0):
    # shortcut: single-user loopback server; use a production server and authentication before network sharing.
    server = ReviewServer(("127.0.0.1", port), ReviewHandler)
    server.runs_dir = runs_dir
    server.token = secrets.token_urlsafe(32)
    server.project_root = Path(__file__).resolve().parent
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        parser.error(f"Runs directory not found: {args.runs_dir}")
    sync_saved_reviews(args.runs_dir)
    with review_server(args.runs_dir, args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Review in your browser: {url}\nPress Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Saved reviews are preserved.")


if __name__ == "__main__":
    main()
