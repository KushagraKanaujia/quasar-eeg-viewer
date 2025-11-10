# Usage Guide

This guide provides detailed instructions for using the EEG/ECG Multichannel Viewer.

## Table of Contents

- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Data Preparation](#data-preparation)
- [Visualization Features](#visualization-features)
- [Python API](#python-api)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KushagraKanaujia/quasar-eeg-viewer.git
cd quasar-eeg-viewer

# Install dependencies
pip install -r requirements.txt
```

### Basic Visualization

```bash
# Visualize your data
python eeg_ecg_plotter.py --data your_data.csv --output visualization.html

# Open the generated HTML file in your browser
open visualization.html  # macOS
xdg-open visualization.html  # Linux
start visualization.html  # Windows
```

## Command-Line Interface

### Basic Usage

```bash
python eeg_ecg_plotter.py [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--data` | `-d` | Path to CSV data file | `EEG and ECG data_02_raw.csv` |
| `--output` | `-o` | Output HTML filename | `eeg_ecg_plot.html` |
| `--help` | `-h` | Show help message | - |

### Examples

```bash
# Use default settings
python eeg_ecg_plotter.py

# Specify custom data file
python eeg_ecg_plotter.py --data recordings/patient_001.csv

# Custom output filename
python eeg_ecg_plotter.py -d data.csv -o analysis.html

# Batch processing
for file in data/*.csv; do
    basename=$(basename "$file" .csv)
    python eeg_ecg_plotter.py -d "$file" -o "output/${basename}_plot.html"
done
```

## Data Preparation

### Required CSV Format

Your CSV file must include:

1. **Time column**: Contains timestamps in seconds
2. **Signal columns**: Channel data in numeric format
3. **Header row**: Column names identifying each channel

### Example Structure

```csv
Time,Fz,Cz,P3,C3,F3,F4,C4,P4,X1:LEOG,X2:REOG,CM
0.0000,12.5,8.3,15.2,9.1,11.4,10.8,7.6,13.2,250.3,245.1,1200.5
0.0033,13.1,8.7,15.8,9.5,11.9,11.2,8.0,13.7,251.2,246.0,1205.3
0.0067,12.8,8.5,16.1,9.3,11.7,11.0,7.8,13.5,250.8,245.6,1202.8
```

### Supported Channel Names

#### EEG Channels (Standard 10-20 System)
- Frontal: `Fz`, `F3`, `F4`, `F7`, `F8`, `Fp1`, `Fp2`
- Central: `Cz`, `C3`, `C4`
- Temporal: `T3`, `T4`, `T5`, `T6`
- Parietal: `Pz`, `P3`, `P4`
- Occipital: `O1`, `O2`
- Auricular: `A1`, `A2`

#### ECG/EOG Channels
- Any channel containing "EOG" or "ECG" in the name
- Examples: `X1:LEOG`, `X2:REOG`, `ECG1`, `ECG2`

#### Reference Channels
- `CM` (Common Mode)
- `REF` (Reference)

### Comment Lines

Lines starting with `#` are automatically skipped:

```csv
# Patient ID: 12345
# Recording Date: 2024-09-25
# Notes: Resting state with eyes closed
Time,Fz,Cz,Pz
0.000,12.5,8.3,15.2
```

### Data Quality Tips

1. **Numeric Format**: Ensure all signal values are numeric (no text)
2. **Consistent Sampling**: Use uniform time intervals
3. **No Missing Values**: Fill gaps or handle NaN values before visualization
4. **Reasonable Ranges**:
   - EEG: typically 10-100 µV
   - ECG: typically 0.1-5 mV
5. **File Size**: For files >100MB, consider downsampling or time windowing

## Visualization Features

### Interactive Controls

Once you open the HTML file in a browser, you can:

#### Pan and Zoom
- **Click and drag**: Pan horizontally through time
- **Scroll wheel**: Zoom in/out
- **Double-click**: Reset zoom to default view
- **Box select**: Click and drag to zoom to specific region

#### Range Slider
- Located below the main plot
- Drag handles to adjust visible time window
- Drag middle section to pan while maintaining zoom level

#### Channel Selection
- **Click legend item**: Toggle individual channel visibility
- **Double-click legend item**: Isolate single channel (hide all others)
- **Double-click again**: Restore all channels

#### Hover Information
- Hover over any point to see:
  - Exact timestamp
  - Channel name
  - Signal amplitude
- Unified crosshair shows values across all visible channels

#### Export
- Click camera icon in toolbar
- Saves current view as high-resolution PNG (1200×700 px)
- Filename: `eeg_ecg_plot.png`

### Understanding the Display

#### Y-Axes

**Left Axis (Primary)**: EEG Channels
- Units: µV (microvolts)
- Typical range: 10-100 µV
- Displayed with solid lines
- Color-coded by channel

**Right Axis (Secondary)**: ECG/Reference Channels
- Units: mV (millivolts)
- Converted from µV (÷1000)
- ECG: dashed lines
- Reference: dotted gray line

#### Legend Organization

Channels are grouped by type:
- **EEG Channels (µV)**: Standard 10-20 positions
- **ECG Channels (mV)**: EOG/ECG signals
- **Reference**: Common-mode reference

#### Color Coding

EEG channels use a consistent color palette:
- Frontal (blue tones)
- Central (orange tones)
- Temporal (green tones)
- Parietal (red tones)
- Occipital (purple tones)

## Python API

### Using as a Library

```python
from eeg_ecg_plotter import load_eeg_data, classify_channels, create_interactive_plot

# Load data
df = load_eeg_data("path/to/data.csv")

# Classify channels
channels = classify_channels(df)

# Create visualization
fig = create_interactive_plot(df, channels, "output.html")
```

### Function Reference

#### `load_eeg_data(filepath)`

Load EEG/ECG data from CSV file.

**Parameters:**
- `filepath` (str): Path to CSV file

**Returns:**
- `pandas.DataFrame`: Loaded data

**Example:**
```python
df = load_eeg_data("recordings/patient_001.csv")
print(f"Loaded {len(df)} samples across {len(df.columns)} channels")
```

#### `classify_channels(df)`

Automatically classify channels into EEG, ECG, and reference groups.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with signal data

**Returns:**
- `dict`: Dictionary with keys 'eeg', 'ecg', 'reference', 'ignored'

**Example:**
```python
channels = classify_channels(df)
print(f"EEG: {channels['eeg']}")
print(f"ECG: {channels['ecg']}")
```

#### `create_interactive_plot(df, channels, output_file='eeg_ecg_plot.html')`

Generate interactive Plotly visualization.

**Parameters:**
- `df` (pandas.DataFrame): Signal data
- `channels` (dict): Channel classifications
- `output_file` (str): Output HTML filename

**Returns:**
- `plotly.graph_objects.Figure`: Plotly figure object

**Example:**
```python
fig = create_interactive_plot(df, channels, "visualization.html")
```

## Advanced Usage

### Custom Channel Classification

Override automatic classification:

```python
from eeg_ecg_plotter import load_eeg_data, create_interactive_plot

df = load_eeg_data("data.csv")

# Custom channel groups
custom_channels = {
    'eeg': ['Fz', 'Cz', 'Pz'],
    'ecg': ['ECG1'],
    'reference': ['REF'],
    'ignored': ['Time', 'Trigger']
}

fig = create_interactive_plot(df, custom_channels, "custom.html")
```

### Time Windowing

Focus on specific time intervals:

```python
df = load_eeg_data("data.csv")

# Extract 30-60 second window
df_window = df[(df['Time'] >= 30) & (df['Time'] <= 60)].copy()
df_window['Time'] = df_window['Time'] - 30  # Reset to start at 0

channels = classify_channels(df_window)
create_interactive_plot(df_window, channels, "window_30_60s.html")
```

### Channel Subsetting

Visualize specific channels:

```python
df = load_eeg_data("data.csv")

# Keep only midline channels
midline = ['Time', 'Fz', 'Cz', 'Pz']
df_midline = df[midline]

channels = classify_channels(df_midline)
create_interactive_plot(df_midline, channels, "midline_only.html")
```

### Batch Processing

Process multiple files programmatically:

```python
import os
from pathlib import Path

input_dir = Path("data")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for csv_file in input_dir.glob("*.csv"):
    df = load_eeg_data(str(csv_file))
    channels = classify_channels(df)

    output_file = output_dir / f"{csv_file.stem}_plot.html"
    create_interactive_plot(df, channels, str(output_file))

    print(f"Processed: {csv_file.name}")
```

## Troubleshooting

### Common Issues

#### Error: "Data file not found"
**Solution:** Check file path is correct and file exists
```bash
ls -l your_data.csv  # Verify file exists
python eeg_ecg_plotter.py --data "$(pwd)/your_data.csv"  # Use absolute path
```

#### Error: "KeyError: 'Time'"
**Solution:** Ensure CSV has a 'Time' column
```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns)  # Check column names
```

#### Visualization shows no data
**Causes:**
- All values are NaN
- Time range is outside data bounds
- Channels misclassified

**Solution:**
```python
df = load_eeg_data("data.csv")
print(df.describe())  # Check data statistics
print(df.isna().sum())  # Check for missing values
```

#### HTML file won't open
**Solution:**
- Try different browser (Chrome, Firefox recommended)
- Check file wasn't corrupted: verify file size >10KB
- Open browser console (F12) to check for JavaScript errors

#### Performance issues with large files
**Solutions:**
1. Downsample data:
```python
df = load_eeg_data("large_file.csv")
df_downsampled = df[::10]  # Keep every 10th sample
```

2. Process in chunks:
```python
chunk_size = 300000  # 1000 seconds at 300 Hz
for i in range(0, len(df), chunk_size):
    chunk = df[i:i+chunk_size]
    create_interactive_plot(chunk, channels, f"chunk_{i}.html")
```

### Getting Help

If you encounter issues:

1. Check existing [GitHub Issues](https://github.com/KushagraKanaujia/quasar-eeg-viewer/issues)
2. Review [examples](../examples/README.md) for similar use cases
3. Open a new issue with:
   - Error message (complete traceback)
   - Sample data (if possible)
   - Python version and OS
   - Library versions: `pip list | grep -E "pandas|plotly|numpy"`

## Best Practices

1. **Data Validation**: Always inspect data before visualization
   ```python
   df = load_eeg_data("data.csv")
   assert not df['Time'].isna().any(), "Time column has missing values"
   assert (df['Time'].diff()[1:] > 0).all(), "Time must be monotonically increasing"
   ```

2. **Memory Management**: For large datasets, process in chunks

3. **Naming Conventions**: Use standard EEG nomenclature for automatic classification

4. **Version Control**: Keep track of processing parameters and versions

5. **Documentation**: Add comments to your CSV files describing the recording
