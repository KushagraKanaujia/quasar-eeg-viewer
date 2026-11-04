# Quasar EEG Viewer

**Turn a raw EEG/ECG recording (CSV) into an interactive plot you can pan, zoom and explore in the browser.**

[![Live demo](https://img.shields.io/badge/live%20demo-netlify-00C7B7?logo=netlify&logoColor=white)](https://kush-quasar-eeg-viewer.netlify.app)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/KushagraKanaujia/quasar-eeg-viewer?quickstart=1)
[![CI](https://github.com/KushagraKanaujia/quasar-eeg-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/KushagraKanaujia/quasar-eeg-viewer/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Try it in one click

| | What you get |
|---|---|
| **[Live demo](https://kush-quasar-eeg-viewer.netlify.app)** | The included 24-channel sample recording, already plotted. You can also open your own CSV; it is read in your browser and never uploaded. The page also links to the HTML files the Python CLI produced from the same sample ([overlay](https://kush-quasar-eeg-viewer.netlify.app/demo/overlay.html), [stacked](https://kush-quasar-eeg-viewer.netlify.app/demo/stacked.html)). |
| **[Open in Codespaces](https://codespaces.new/KushagraKanaujia/quasar-eeg-viewer?quickstart=1)** | A ready Python 3.12 environment. It installs the package, runs the CLI on the sample data and opens the demo page on port 8000. After that you can run `quasar-eeg-viewer` and `pytest` in the terminal. |

## What it does

EEG headsets record many channels at once. Brain channels (EEG) are tens of microvolts, while eye and heart channels (EOG/ECG) and the common-mode reference are often thousands of times larger. If you put them all on one axis, the brain signals look flat.

Quasar EEG Viewer reads a CSV export, works out what kind of signal each column is from its name, and writes a single self-contained HTML file with:

- **Two y-axes**: EEG in µV on the left; ECG/EOG and reference converted to mV on the right
- **Pan, zoom and a range slider**, starting with a 30-second window
- **Legend toggles**, grouped by EEG / ECG-EOG / reference
- **Unified hover** that shows every channel's value at one time point
- **PNG export** and drawing tools in the toolbar
- **A stacked view** (`--subplot`) with one row per channel

The bundled sample (`EEG and ECG data_02_raw.csv`) is a 42.5 s, 300 Hz export from a DSI-24 headset with 21 EEG channels, 2 EOG channels and the CM reference.

## How it works

```
CSV ──► EEGDataLoader ──► ChannelClassifier ──► EEGPlotter ──► interactive .html
        skip "#" lines     EEG / ECG / ref /     Plotly figure,
        need a Time col    ignored / unknown     dual y-axis
```

1. **Load** (`core/data_loader.py`): drops comment lines starting with `#` (device header lines), parses the rest with pandas and checks that there is a `Time` column.
2. **Classify** (`core/channel_classifier.py`): matches column names to lists in `core/config.py`. 10-20 electrode names (`Fz`, `Cz`, `P3`, …) are EEG, `X1:LEOG`/`X2:REOG` are ECG/EOG, `CM` is the reference, and device columns (`Trigger`, `ADC_Status`, …) are ignored.
3. **Plot** (`visualization/plotter.py`): builds the Plotly figure and writes HTML. plotly.js is embedded by default so the file works offline. Set `visualization_config.include_plotlyjs = "cdn"` to get much smaller files.

The live demo page (`site/index.html`) repeats the same three steps in JavaScript (Papa Parse + Plotly.js), so you can try it without installing Python. One difference: columns it does not recognise are still plotted, but they stay hidden until you click them in the legend.

## Tech stack

| | |
|---|---|
| Core | Python 3.8+, [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Plotly](https://plotly.com/python/) |
| CLI | `argparse` entry point `quasar-eeg-viewer` |
| Tests | pytest + pytest-cov (`tests/`), run by GitHub Actions on Python 3.10 and 3.12 |
| Demo site | Static HTML with [Pico CSS](https://picocss.com/), [Papa Parse](https://www.papaparse.com/), [Plotly.js](https://plotly.com/javascript/), hosted on Netlify |
| Dev environment | Dev container based on the [devcontainers/templates](https://github.com/devcontainers/templates) Python image |

## Run locally

```bash
git clone https://github.com/KushagraKanaujia/quasar-eeg-viewer.git
cd quasar-eeg-viewer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

quasar-eeg-viewer                       # plots the bundled sample -> eeg_ecg_plot.html
quasar-eeg-viewer --data my.csv -o my.html
quasar-eeg-viewer --data my.csv --subplot
quasar-eeg-viewer --data my.csv --info  # file size and column names only
pytest                                  # run the tests
```

Open the HTML file in any browser.

**Build the demo site** (the same files that are on Netlify):

```bash
python scripts/build_demo.py
python -m http.server 8000 --directory site   # http://localhost:8000
```

### Python API

```python
from quasar_eeg import quick_plot
fig = quick_plot("data.csv", "output.html")
```

```python
from quasar_eeg import EEGDataLoader, ChannelClassifier, EEGPlotter

df = EEGDataLoader().load_from_file("data.csv")
channels = ChannelClassifier().classify(df)   # {'eeg': [...], 'ecg': [...], 'reference': [...], ...}
fig = EEGPlotter().create_plot(df, channels, "output.html")
```

To add a channel name the classifier does not know, use `ChannelClassifier().add_custom_channel("EKG", "ecg")`. More detail is in [docs/USAGE.md](docs/USAGE.md) and [docs/API.md](docs/API.md).

### Input format

- One header row that includes `Time` (seconds), followed by one row per sample
- Any number of `#` comment lines before the header (DSI-Streamer exports work as they are)
- Signal values in µV

```csv
# Sample_Frequency_(Hz) =,300
Time,Fz,Cz,P3,X1:LEOG,X2:REOG,CM
0.0,12.5,8.3,15.2,250.3,245.1,1200.5
0.0033,13.1,8.7,15.8,251.2,246.0,1205.3
```

## Project layout

```
src/quasar_eeg/      package: cli.py, core/ (config, loader, classifier), visualization/, utils/
tests/               pytest suite
scripts/build_demo.py  builds site/demo/*.html and site/sample.csv with the real package
site/index.html      static demo page (in-browser viewer)
.devcontainer/       Codespaces / Dev Containers setup
eeg_ecg_plotter.py   original single-file script, kept for backward compatibility
```

## Ideas for later

Filtering (high-pass, low-pass, notch), spectral plots, annotations and EDF/BIDS import are not implemented yet. Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

- [Plotly](https://github.com/plotly/plotly.py) and [Plotly.js](https://github.com/plotly/plotly.js) (MIT): interactive charts
- [pandas](https://github.com/pandas-dev/pandas) (BSD-3-Clause) and [NumPy](https://github.com/numpy/numpy) (BSD-3-Clause): data handling
- [Papa Parse](https://github.com/mholt/PapaParse) (MIT): CSV parsing in the browser demo
- [Pico CSS](https://github.com/picocss/pico) (MIT): demo page styling
- [devcontainers/templates](https://github.com/devcontainers/templates) (MIT): base for the Codespaces config
- [shields.io](https://shields.io) badges
- Electrode names follow the international 10-20 system

## License

MIT, see [LICENSE](LICENSE).
