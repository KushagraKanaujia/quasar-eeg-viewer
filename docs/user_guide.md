# QUASAR EEG Viewer - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Loading Data](#loading-data)
4. [Visualization Features](#visualization-features)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### First Time Setup

1. **Run the Setup Script**
   ```bash
   # On Unix/macOS
   chmod +x scripts/setup.sh
   ./scripts/setup.sh

   # On Windows
   scripts\setup.bat
   ```

2. **Verify Installation**
   The setup script will generate sample data and test the installation.

3. **Start the Application**
   ```bash
   source venv/bin/activate  # Activate virtual environment
   python src/eeg_viewer.py
   ```

4. **Access the Web Interface**
   Open your browser to `http://127.0.0.1:8050`

## Interface Overview

The QUASAR EEG Viewer interface consists of several key components:

### Control Panel (Top Section)
- **Data File Upload**: Drag-and-drop area for CSV files
- **Time Window Slider**: Adjusts the duration of data shown (1-30 seconds)
- **Start Time Slider**: Controls which part of the data to display

### Channel Selection (Middle Section)
- **EEG Channels**: Checkbox list for EEG signal channels
- **ECG Channels**: Checkbox list for ECG signal channels

### Export Controls
- **Export Current View**: Saves the currently visible data window
- **Export Full Data**: Saves the complete loaded dataset

### Main Visualization (Center)
- **Interactive Plot**: Plotly-based chart with zoom, pan, and hover features
- **Dual Y-Axes**: Left axis for EEG (µV), right axis for ECG (mV)
- **Legend**: Click to toggle individual traces on/off

### Statistics Panel (Bottom)
- **Real-time Statistics**: Mean, std dev, min/max for selected channels
- **Sample Counts**: Number of data points in current view

## Loading Data

### Supported File Formats
The application currently supports CSV files with the following requirements:

#### File Structure
```csv
Time,EEG_C3,EEG_C4,EEG_O1,EEG_O2,ECG_I,ECG_II,ECG_III
0.000,0.000012,-0.000008,0.000015,-0.000003,0.0012,0.0008,0.0015
0.001,0.000015,-0.000012,0.000018,-0.000006,0.0015,0.0012,0.0018
...
```

#### Column Requirements
- **Time Column**: Must be named `Time`, `time`, or `timestamp`
  - Should be in seconds
  - Must be monotonically increasing
  - If missing, will be auto-generated based on sampling rate

- **Channel Columns**:
  - EEG channels should contain "EEG" in name or use standard electrode names (C3, C4, O1, O2, F3, F4, P3, P4, etc.)
  - ECG channels should contain "ECG" in name or use lead names (I, II, III, aVR, aVL, aVF, V1-V6)

#### Data Units
- **EEG Data**: Should be in volts (will be converted to µV for display)
- **ECG Data**: Should be in volts (will be converted to mV for display)
- **Typical Ranges**:
  - EEG: ±100 µV (±100e-6 V in file)
  - ECG: ±5 mV (±5e-3 V in file)

### Loading Methods

#### 1. Drag and Drop
1. Simply drag your CSV file into the upload area
2. The file will be processed automatically
3. Channels will be detected and classified

#### 2. Command Line
```bash
python src/eeg_viewer.py --data path/to/your/data.csv
```

#### 3. Sample Data
Use the provided sample data to test the application:
```bash
python src/eeg_viewer.py --data data/sample_eeg_data.csv
```

## Visualization Features

### Interactive Controls

#### Zooming
- **Mouse Wheel**: Zoom in/out on the plot
- **Zoom Box**: Click and drag to select an area to zoom into
- **Double Click**: Reset zoom to fit all data

#### Panning
- **Click and Drag**: Pan across the time axis
- **Time Sliders**: Use the start time slider for precise navigation

#### Hover Information
- Move your mouse over any trace to see:
  - Exact time value
  - Signal amplitude
  - Channel name

### Dual-Axis Scaling

The application automatically handles the different scales of EEG and ECG signals:

- **Left Y-Axis (EEG)**: Scaled in microvolts (µV)
- **Right Y-Axis (ECG)**: Scaled in millivolts (mV)

This prevents ECG signals from overwhelming the smaller EEG signals in the visualization.

### Channel Management

#### Selecting Channels
- Use the checkboxes to select which channels to display
- Changes are applied immediately
- You can select any combination of EEG and ECG channels

#### Color Coding
- **EEG Channels**: Blue colors by default
- **ECG Channels**: Red colors by default
- Colors can be customized in the configuration file

### Time Window Control

#### Window Size
- Adjust from 1 to 30 seconds
- Smaller windows show more detail
- Larger windows show more context

#### Navigation
- Use the start time slider to scroll through the data
- The slider range adapts to your data length
- Time stamps are shown in the slider tooltip

## Advanced Usage

### Configuration File

Create a `config/viewer_config.json` file to customize the application:

```json
{
    "sampling_rate": 1000,
    "eeg_scale_factor": 1e-6,
    "ecg_scale_factor": 1e-3,
    "window_size": 10.0,
    "eeg_channels": ["EEG_C3", "EEG_C4", "EEG_O1", "EEG_O2"],
    "ecg_channels": ["ECG_I", "ECG_II", "ECG_III"],
    "colors": {
        "eeg": "#1f77b4",
        "ecg": "#d62728",
        "background": "#ffffff",
        "grid": "#f0f0f0"
    }
}
```

#### Configuration Options
- `sampling_rate`: Default sampling rate in Hz
- `eeg_scale_factor`: Conversion factor for EEG data
- `ecg_scale_factor`: Conversion factor for ECG data
- `window_size`: Default time window in seconds
- `eeg_channels`: List of expected EEG channel names
- `ecg_channels`: List of expected ECG channel names
- `colors`: Color scheme customization

### Command Line Options

```bash
python src/eeg_viewer.py [OPTIONS]

Options:
  --data, -d PATH     CSV data file path
  --config, -c PATH   Configuration JSON file path
  --host TEXT         Host address (default: 127.0.0.1)
  --port INTEGER      Port number (default: 8050)
  --debug             Run in debug mode
  --help              Show help message
```

#### Examples
```bash
# Load specific data file
python src/eeg_viewer.py --data my_eeg_data.csv

# Use custom configuration
python src/eeg_viewer.py --config my_config.json

# Run on different port
python src/eeg_viewer.py --port 8080

# Debug mode with verbose logging
python src/eeg_viewer.py --debug
```

### Data Export

#### Export Current View
- Exports only the data currently visible in the plot
- Includes selected time window and channels
- Filename includes timestamp: `eeg_export_view_YYYYMMDD_HHMMSS.csv`

#### Export Full Data
- Exports the complete loaded dataset
- Includes all time points and selected channels
- Filename includes timestamp: `eeg_export_full_YYYYMMDD_HHMMSS.csv`

#### Export File Format
Exported files maintain the same CSV format as input:
```csv
Time,EEG_C3,EEG_C4,ECG_I
0.000,0.000012,-0.000008,0.0012
0.001,0.000015,-0.000012,0.0015
...
```

### Performance Optimization

#### For Large Datasets
1. **Reduce Time Window**: Use smaller time windows (5-10 seconds)
2. **Select Fewer Channels**: Display only necessary channels
3. **Use Data Decimation**: For overview, consider pre-processing data
4. **Close Browser Tabs**: Free up memory by closing unused tabs

#### Hardware Requirements
- **RAM**: 4GB minimum, 8GB recommended for large datasets
- **CPU**: Modern multi-core processor recommended
- **Browser**: Chrome, Firefox, or Safari (latest versions)

## Troubleshooting

### Common Issues

#### Application Won't Start
**Problem**: `python src/eeg_viewer.py` fails
**Solutions**:
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (3.8+ required)

#### Data Not Loading
**Problem**: CSV file upload fails or shows no data
**Solutions**:
1. Check file format - ensure CSV with proper headers
2. Verify time column exists and is named correctly
3. Check for empty or corrupted files
4. Ensure file size is reasonable (<100MB for web upload)

#### Channels Not Detected
**Problem**: EEG/ECG channels don't appear in selection lists
**Solutions**:
1. Check channel naming conventions
2. Add custom channel names to configuration file
3. Rename columns to include "EEG" or "ECG" prefixes

#### Performance Issues
**Problem**: Slow loading or laggy interface
**Solutions**:
1. Reduce time window size
2. Select fewer channels
3. Use smaller datasets for testing
4. Close other browser tabs
5. Restart the application

#### Port Already in Use
**Problem**: `Address already in use` error
**Solutions**:
1. Use different port: `--port 8051`
2. Stop other applications using port 8050
3. Restart your computer if needed

### Browser Compatibility

#### Supported Browsers
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

#### Known Issues
- Internet Explorer is not supported
- Some mobile browsers may have limited functionality
- Ad blockers may interfere with the interface

### Getting Help

If you encounter issues not covered here:

1. **Check the main README.md** for additional information
2. **Review the troubleshooting.md** file in the docs folder
3. **Enable debug mode** to see detailed error messages:
   ```bash
   python src/eeg_viewer.py --debug
   ```
4. **Contact support** with the following information:
   - Operating system and version
   - Python version
   - Error messages from the console
   - Sample data file (if possible)

### Tips for Best Experience

1. **Start Small**: Begin with the provided sample data
2. **Test Configuration**: Verify settings with known good data
3. **Monitor Performance**: Watch browser memory usage with large files
4. **Save Work**: Export important views before loading new data
5. **Keep Backups**: Maintain copies of your original data files