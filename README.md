# QUASAR EEG/ECG Multichannel Plotter

**Scrollable/zoomable interactive visualization of EEG and ECG signals with proper scaling**

This project implements the QUASAR coding screener requirements for a multichannel EEG/ECG plotting tool that handles the provided dataset with appropriate scaling and interactive navigation.

## How to Run

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default QUASAR dataset
python eeg_ecg_plotter.py

# Or specify custom data file
python eeg_ecg_plotter.py --data "EEG and ECG data_02_raw.csv" --output my_plot.html
```

### Output
- Generates an interactive HTML file (`eeg_ecg_plot.html` by default)
- Open the HTML file in any web browser to explore the data
- No server required - completely standalone visualization

## Design Choices

### EEG vs ECG Scaling Strategy

The key challenge was making both EEG (µV) and ECG (mV) signals visible on the same plot despite their vastly different amplitude ranges.

**Solution: Dual Y-Axis Approach**
- **Primary Y-axis (left)**: EEG channels in µV (10s-100s range)
  - Channels: Fz, Cz, P3, C3, F3, F4, C4, P4, Fp1, Fp2, T3, T4, T5, T6, O1, O2, F7, F8, A1, A2, Pz
  - Data displayed as-is (already in µV in the dataset)

- **Secondary Y-axis (right)**: ECG channels in mV (converted from µV)
  - Channels: X1:LEOG (Left ECG), X2:REOG (Right ECG)
  - Data converted from µV to mV by dividing by 1000
  - Displayed with dashed lines to differentiate from EEG

- **Reference Channel**: CM (Common Mode) plotted on secondary axis in mV
  - Large amplitude reference signal, converted to mV for consistency
  - Displayed as dotted gray line with reduced opacity

### Interactive Features
- **Range Slider**: Built-in Plotly range slider for easy scrolling through time
- **Pan/Zoom**: Mouse-based navigation with zoom controls
- **Channel Toggle**: Click legend items to show/hide individual channels
- **Hover Info**: Unified crossfilter showing values across all visible channels
- **Export**: Built-in PNG export functionality via Plotly toolbar

### Data Processing
- Automatically skips comment lines starting with `#` as specified
- Identifies channels based on QUASAR naming conventions
- Ignores non-signal columns (Trigger, Time_Offset, ADC_Status, etc.)
- Preserves full temporal resolution for accurate signal representation

## Technical Implementation

### Core Libraries
- **Plotly**: Interactive plotting with built-in pan/zoom/scroll capabilities
- **Pandas**: Efficient data loading and CSV parsing
- **NumPy**: Numerical operations and data type handling

### Key Functions
- `load_quasar_data()`: Parses CSV while skipping comment lines
- `classify_channels()`: Automatically categorizes EEG/ECG/reference channels
- `create_interactive_plot()`: Generates dual-axis interactive visualization

### Performance Optimizations
- Efficient data loading with comment line filtering
- Optimized Plotly configuration for smooth interaction
- Grouped legends for better organization
- Default time window limiting for initial display

## AI Assistance Disclosure

This project was developed with strategic AI assistance to meet the 2-4 hour timeline while ensuring high code quality:

### Human-Led Design Decisions
- **Architecture choice**: Single-script approach for simplicity and portability
- **Scaling strategy**: Dual y-axis solution after analyzing amplitude differences
- **Technology selection**: Plotly for robust interactive capabilities
- **Feature prioritization**: Focus on core scrolling/zooming requirements
- **User experience**: Legend grouping and intuitive navigation design

### AI-Accelerated Implementation
- **Boilerplate code generation**: Argument parsing, file I/O, error handling
- **Plotly configuration**: Interactive features and layout optimization
- **Data processing logic**: CSV parsing and channel classification
- **Code documentation**: Function docstrings and inline comments

### Technical Validation
- Tested with actual QUASAR dataset to verify scaling correctness
- Validated channel detection against specification requirements
- Confirmed interactive features meet usability expectations

This approach demonstrates effective human-AI collaboration where domain expertise guides architectural decisions while AI accelerates implementation of well-defined requirements.

## Features Delivered

✅ **Core Requirements Met:**
- ✅ Loads QUASAR CSV data (skipping # comment lines)
- ✅ Interactive scrolling/panning/zooming via Plotly
- ✅ Dual-axis scaling (EEG µV, ECG mV) for proper visibility
- ✅ Channel selection via legend toggles
- ✅ Standalone HTML output (no server required)

✅ **Enhanced Usability:**
- ✅ Built-in range slider for quick navigation
- ✅ Grouped legend organization (EEG/ECG/Reference)
- ✅ Hover tooltips with precise values
- ✅ Export functionality (PNG via toolbar)
- ✅ Command-line interface with help
- ✅ Automatic channel classification

## Future Work

Given more time, the following enhancements would improve the tool:

### Signal Processing Features
- **Filtering options**: High-pass, low-pass, and notch filters for artifact removal
- **Normalization**: Per-channel amplitude normalization for comparison
- **Spectral analysis**: Power spectral density plots for frequency domain analysis
- **Annotation tools**: Ability to mark and label signal events

### User Interface Improvements
- **Time range selection**: Input boxes for precise time window specification
- **Amplitude scaling controls**: Manual y-axis range adjustment
- **Multi-file support**: Load and compare multiple recordings
- **Configuration presets**: Save/load channel selections and display settings

### Advanced Visualizations
- **Topographic maps**: 2D spatial representation of EEG activity
- **Spectrogram view**: Time-frequency heatmaps for signal analysis
- **Statistical overlays**: Mean, standard deviation bands
- **Cross-correlation analysis**: Inter-channel relationship visualization

### Export and Analysis
- **Data export**: Selected time windows as CSV/MAT files
- **Report generation**: Automated analysis reports with statistics
- **Batch processing**: Command-line tools for multiple file analysis
- **Integration APIs**: Hooks for external analysis tools

---

**Time Investment**: ~3 hours total (within specified 2-4 hour range)
- 1 hour: Requirements analysis and technical design
- 1.5 hours: Implementation and testing with real data
- 0.5 hours: Documentation and final polish