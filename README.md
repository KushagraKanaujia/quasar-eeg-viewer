# Quasar EEG Viewer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Visualize brain waves and heart signals in your browser - no complicated setup required.**

Quasar EEG Viewer turns raw EEG (brain activity) and ECG (heart rate) data into interactive web visualizations. If you work with neural signals, sleep studies, brain-computer interfaces, or cardiac monitoring, this tool makes exploring your data easier.

## Features

### Interactive Visualization
- **Pan & Zoom**: Smooth navigation through time-series data with mouse-based controls
- **Range Slider**: Quick temporal navigation with built-in scrollbar
- **Channel Toggling**: Click legend items to show/hide individual channels
- **Export Capabilities**: High-resolution PNG export via integrated toolbar
- **Unified Hover**: Crosshair display showing synchronized values across all channels

### Intelligent Signal Scaling
- **Dual Y-Axis System**: Separate scaling for EEG (µV) and ECG (mV) signals
- **Automatic Channel Classification**: Smart detection of signal types based on naming conventions
- **Reference Channel Support**: Dedicated handling of common-mode reference signals
- **Visual Differentiation**: Distinct line styles (solid/dashed/dotted) for different signal types

### Data Processing
- **Flexible Input**: Supports CSV format with automatic comment line filtering
- **High Performance**: Efficient handling of large datasets with optimized rendering
- **Standard Compliance**: Compatible with common EEG/ECG data formats
- **Robust Parsing**: Handles various CSV structures and naming conventions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install

Install from the repository:

```bash
git clone https://github.com/KushagraKanaujia/quasar-eeg-viewer.git
cd quasar-eeg-viewer
pip install -e .
```

For development (with testing and linting tools):

```bash
pip install -e ".[dev]"
```

### Using pip (once published to PyPI)

```bash
pip install quasar-eeg-viewer
```

## Usage

### Command-Line Interface

After installation, use the `quasar-eeg-viewer` command:

```bash
# Basic usage with default data file
quasar-eeg-viewer

# Specify custom data file
quasar-eeg-viewer --data path/to/your/data.csv --output analysis.html

# View file information without plotting
quasar-eeg-viewer --data data.csv --info

# Create subplot view
quasar-eeg-viewer --data data.csv --subplot

# Adjust logging level
quasar-eeg-viewer --data data.csv --log-level DEBUG
```

### Python API

Use the package programmatically in your Python code:

```python
from quasar_eeg import quick_plot

# Quick plotting
fig = quick_plot('data.csv', 'output.html')
```

For more control:

```python
from quasar_eeg import EEGDataLoader, ChannelClassifier, EEGPlotter

# Load data
loader = EEGDataLoader()
df = loader.load_from_file('data.csv')

# Classify channels
classifier = ChannelClassifier()
channels = classifier.classify(df)

# Create visualization
plotter = EEGPlotter()
fig = plotter.create_plot(df, channels, 'output.html')
```

### Legacy Script (Backward Compatibility)

The original script is still available for backward compatibility:

```bash
python eeg_ecg_plotter.py --data data.csv --output output.html
```

## Data Format

The tool expects CSV files with:
- **Time column**: Timestamps in seconds
- **Signal columns**: Named channels (e.g., Fz, Cz, P3, X1:LEOG, etc.)
- **Comment support**: Lines starting with `#` are automatically skipped
- **Standard 10-20 system**: EEG electrode naming convention

Example structure:
```csv
Time,Fz,Cz,P3,C3,F3,F4,C4,P4,X1:LEOG,X2:REOG,CM
0.0,12.5,8.3,15.2,9.1,11.4,10.8,7.6,13.2,250.3,245.1,1200.5
0.0033,13.1,8.7,15.8,9.5,11.9,11.2,8.0,13.7,251.2,246.0,1205.3
```

## Technical Architecture

### Core Technologies
| Library | Purpose | Version |
|---------|---------|---------|
| **Plotly** | Interactive visualization and WebGL rendering | 5.15.0+ |
| **Pandas** | Efficient data manipulation and CSV parsing | 1.5.0+ |
| **NumPy** | Numerical computations and array operations | 1.21.0+ |

### Architecture Highlights
- **Modular package structure**: Clean separation of concerns across modules
- **Standalone HTML output**: No server required for visualization
- **Comprehensive error handling**: Robust DataLoadError and PlotterError exceptions
- **Configurable logging**: Built-in logging system for debugging and monitoring
- **Memory efficient**: Streaming data processing for large files
- **Fully tested**: Comprehensive unit test coverage with pytest
- **Type-safe configuration**: Dataclass-based configuration management
- **CI/CD pipeline**: Automated testing across multiple Python versions and platforms

### Channel Classification Logic
The tool automatically categorizes channels into three groups:

1. **EEG Channels** (Primary Y-axis, µV scale)
   - Standard 10-20 system positions: Fz, Cz, P3, C3, F3, F4, C4, P4, Fp1, Fp2, T3, T4, T5, T6, O1, O2, F7, F8, A1, A2, Pz
   - Displayed with solid lines in color-coded palette

2. **ECG/EOG Channels** (Secondary Y-axis, mV scale)
   - Eye movement and cardiac channels: X1:LEOG, X2:REOG
   - Converted from µV to mV (÷1000) for appropriate scaling
   - Displayed with dashed lines for visual distinction

3. **Reference Channels** (Secondary Y-axis, mV scale)
   - Common-mode reference: CM
   - Displayed with dotted gray line at reduced opacity

## Use Cases

### Clinical Applications
- **Sleep study analysis**: Review polysomnography recordings
- **Seizure detection**: Identify abnormal EEG patterns
- **Cardiac monitoring**: Analyze ECG waveforms alongside brain activity

### Research Applications
- **Signal quality assessment**: Validate recording integrity
- **Artifact identification**: Detect movement and electrical interference
- **Multi-modal analysis**: Compare brain and cardiac activity patterns

### Educational Applications
- **Signal processing instruction**: Demonstrate filtering and scaling concepts
- **Data visualization training**: Teach interactive plotting techniques
- **Neuroscience education**: Explore real physiological signals

## Project Structure

```
quasar-eeg-viewer/
├── src/quasar_eeg/              # Main package source
│   ├── __init__.py              # Package initialization and exports
│   ├── cli.py                   # Command-line interface
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── data_loader.py       # Data loading and validation
│   │   └── channel_classifier.py # Channel classification logic
│   ├── visualization/           # Plotting components
│   │   ├── __init__.py
│   │   └── plotter.py           # Interactive plot generation
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       └── logger.py            # Logging utilities
├── tests/                       # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration & fixtures
│   ├── test_data_loader.py
│   ├── test_channel_classifier.py
│   └── test_plotter.py
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── .github/workflows/           # CI/CD pipelines
│   └── ci.yml                   # GitHub Actions workflow
├── eeg_ecg_plotter.py          # Legacy script (backward compatibility)
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Setup configuration
├── requirements.txt            # Dependencies
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Future Enhancements

### Signal Processing
- Digital filtering (high-pass, low-pass, notch filters)
- Spectral analysis and power spectral density plots
- Independent component analysis (ICA) for artifact removal
- Time-frequency analysis (wavelet transforms, spectrograms)

### User Interface
- Real-time streaming data support
- Multi-file comparison mode
- Annotation and event marking tools
- Configurable color schemes and themes

### Advanced Features
- Automated artifact detection algorithms
- Statistical analysis and reporting
- Integration with neuroimaging data (fMRI, MEG)
- Export to analysis formats (EDF, BIDS)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup
```bash
git clone https://github.com/KushagraKanaujia/quasar-eeg-viewer.git
cd quasar-eeg-viewer
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [Plotly](https://plotly.com/python/) for interactive visualizations
- Data processing with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Follows the International 10-20 system for EEG electrode placement

## Questions?

Open an issue on GitHub if you need help or have suggestions.

---

**Stack**: Python 3.8+ | Plotly | Pandas | NumPy
