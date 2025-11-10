# EEG/ECG Multichannel Viewer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An interactive web-based visualization tool for multi-channel EEG (Electroencephalography) and ECG (Electrocardiography) signal analysis. Built with Python, Plotly, and modern data science libraries to provide researchers and clinicians with powerful signal exploration capabilities.

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
```bash
git clone https://github.com/KushagraKanaujia/quasar-eeg-viewer.git
cd quasar-eeg-viewer
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python eeg_ecg_plotter.py
```

This generates an interactive HTML file (`eeg_ecg_plot.html`) that can be opened in any modern web browser.

### Custom Data File
```bash
python eeg_ecg_plotter.py --data path/to/your/data.csv --output custom_plot.html
```

### Command-Line Options
```
--data, -d    Path to CSV data file (default: EEG and ECG data_02_raw.csv)
--output, -o  Output HTML filename (default: eeg_ecg_plot.html)
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
- **Single-script deployment**: Portable and easy to integrate
- **Standalone HTML output**: No server required for visualization
- **Modular design**: Clear separation of data loading, classification, and plotting
- **Memory efficient**: Streaming data processing for large files

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
├── eeg_ecg_plotter.py          # Main application script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── EEG and ECG data_02_raw.csv # Sample dataset
└── eeg_ecg_plot.html           # Generated output (example)
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

- Built with the excellent [Plotly](https://plotly.com/python/) library for interactive visualizations
- Data processing powered by [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Follows standard EEG electrode placement conventions (International 10-20 system)

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Technical Stack**: Python 3.8+ | Plotly | Pandas | NumPy | HTML5
