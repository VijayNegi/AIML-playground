"""Print within-input-pair Spearman correlations between automatic scores and reviews."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from review_runs import RATINGS


def spearman(values, ratings):
    """Pearson correlation of average ranks, including tied observations."""
    def ranks(items):
        _, inverse, counts = np.unique(items, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    x, y = ranks(values), ranks(ratings)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlations(runs_dir):
    groups = {}
    for review_path in sorted(runs_dir.glob("*/review.json")):
        try:
            folder = review_path.parent
            params = json.loads((folder / "params.json").read_text())
            metrics = json.loads((folder / "metrics.json").read_text())
            review = json.loads(review_path.read_text())
            pair = params["input_pair_id"]
            protocol = (metrics["quality_version"], metrics.get("quality_pyiqa_version", "unavailable"),
                         metrics.get("quality_maniqa_model", "unavailable"),
                         tuple(metrics.get("quality_maniqa_crop_bbox", [])),
                         tuple(tuple(tile["bbox"]) for tile in metrics.get("quality_maniqa_tiles", [])))
            for name, value in metrics.items():
                if not (name in ("quality_maniqa", "quality_maniqa_min") or name.startswith("quality_preservation_") or
                        name.endswith("_gradient_excess") or name.endswith("_delta_e76")):
                    continue
                if type(value) not in (int, float) or not math.isfinite(value):
                    continue
                for rating in RATINGS:
                    if type(review.get(rating)) is int and 1 <= review[rating] <= 5:
                        groups.setdefault((pair, protocol, name, rating), []).append((value, review[rating]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Skipping {review_path.parent}: {error}", file=sys.stderr)
    for (pair, protocol, name, rating), observations in sorted(groups.items()):
        if len(observations) < 3:
            continue
        values, ratings = zip(*observations)
        rho = spearman(values, ratings)
        yield {"input_pair_id": pair, "quality_protocol": json.dumps(protocol), "metric": name,
               "human_rating": rating, "runs": len(observations), "spearman": rho,
               "expected_sign": "positive" if name in ("quality_maniqa", "quality_maniqa_min") else "negative"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        parser.error(f"Runs directory not found: {args.runs_dir}")
    rows = list(correlations(args.runs_dir))
    if not rows:
        print("Need at least three scored and reviewed runs with the same input pair and assessment protocol.", file=sys.stderr)
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print("Exploratory correlations, not accuracy estimates. Blank rho means constant scores/ratings. "
          "Validate promising metrics on held-out image pairs; do not learn weights from this report.", file=sys.stderr)


if __name__ == "__main__":
    main()
