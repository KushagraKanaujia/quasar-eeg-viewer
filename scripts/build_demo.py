#!/usr/bin/env python3
"""
Build the static demo site in ``site/``.

Runs the real quasar_eeg pipeline on the bundled sample recording and writes:

  site/demo/overlay.html   - dual-axis overlay view (what the CLI makes by default)
  site/demo/stacked.html   - one row per channel (the CLI's --subplot view)
  site/sample.csv          - the sample recording, used by the in-browser viewer

The generated files load plotly.js from its CDN so the pages stay small.

Usage:
    python scripts/build_demo.py
    python -m http.server 8000 --directory site   # then open http://localhost:8000
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quasar_eeg import (  # noqa: E402
    ApplicationConfig,
    ChannelClassifier,
    EEGDataLoader,
    EEGPlotter,
)

SAMPLE = ROOT / "EEG and ECG data_02_raw.csv"
SITE = ROOT / "site"
DEMO = SITE / "demo"


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)

    config = ApplicationConfig()
    config.log_level = "WARNING"
    config.visualization_config.include_plotlyjs = "cdn"

    df = EEGDataLoader(config).load_from_file(SAMPLE)
    channels = ChannelClassifier(config).classify(df)
    plotter = EEGPlotter(config)

    plotter.create_plot(df, channels, DEMO / "overlay.html")
    plotter.create_subplot_by_channel(df, channels, DEMO / "stacked.html")
    shutil.copyfile(SAMPLE, SITE / "sample.csv")

    for f in sorted([*DEMO.glob("*.html"), SITE / "sample.csv"]):
        print(f"wrote {f.relative_to(ROOT)} ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
