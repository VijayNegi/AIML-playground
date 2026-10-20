"""Project paths. Data stays at the project root so existing absolute run paths remain valid."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"
SAMPLES_DIR = ROOT / "samples"
RUNS_DIR = ROOT / "runs"
AIM_REPO = ROOT
BATCHES_FILE = ROOT / "batches.json"
