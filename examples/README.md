# Examples

This directory contains example usage and sample outputs for the EEG/ECG Multichannel Viewer.

## Sample Visualizations

### Basic EEG/ECG Visualization

The main example demonstrates visualization of multi-channel EEG data with ECG channels:

```bash
# Generate example visualization
cd ..
python eeg_ecg_plotter.py --data "EEG and ECG data_02_raw.csv" --output examples/sample_output.html
```

### Features Demonstrated

1. **Multi-Channel Display**
   - 21 EEG channels (Fz, Cz, P3, C3, F3, F4, C4, P4, Fp1, Fp2, T3, T4, T5, T6, O1, O2, F7, F8, A1, A2, Pz)
   - 2 ECG/EOG channels (X1:LEOG, X2:REOG)
   - 1 Reference channel (CM)

2. **Dual-Axis Scaling**
   - Primary Y-axis: EEG signals in µV (typical range: 10-100 µV)
   - Secondary Y-axis: ECG/Reference signals in mV (converted from µV)

3. **Interactive Controls**
   - Pan: Click and drag on the plot
   - Zoom: Scroll or use zoom buttons
   - Range Selection: Use the slider below the plot
   - Channel Toggle: Click legend items to show/hide channels
   - Export: Use the camera icon to save as PNG

## Custom Data Examples

### Example 1: Custom Time Window

Focus on a specific time window of interest:

```python
import pandas as pd
from eeg_ecg_plotter import load_eeg_data, classify_channels, create_interactive_plot

# Load data
df = load_eeg_data("your_data.csv")

# Filter to specific time range (e.g., 30-60 seconds)
df_filtered = df[(df['Time'] >= 30) & (df['Time'] <= 60)]

# Create plot
channels = classify_channels(df_filtered)
create_interactive_plot(df_filtered, channels, "custom_window.html")
```

### Example 2: Subset of Channels

Visualize only specific channels:

```python
import pandas as pd
from eeg_ecg_plotter import load_eeg_data, classify_channels, create_interactive_plot

# Load data
df = load_eeg_data("your_data.csv")

# Select specific channels
channels_to_keep = ['Time', 'Fz', 'Cz', 'Pz', 'O1', 'O2', 'X1:LEOG', 'X2:REOG']
df_subset = df[channels_to_keep]

# Create plot
channels = classify_channels(df_subset)
create_interactive_plot(df_subset, channels, "subset_channels.html")
```

### Example 3: Multiple Files Comparison

Process multiple files sequentially:

```bash
# Process multiple recordings
for file in data/*.csv; do
    filename=$(basename "$file" .csv)
    python eeg_ecg_plotter.py --data "$file" --output "examples/${filename}_plot.html"
done
```

## Data Format Examples

### Minimal CSV Format

```csv
Time,Fz,Cz,Pz
0.000,12.5,8.3,15.2
0.003,13.1,8.7,15.8
0.007,12.8,8.5,16.1
```

### With Comments

```csv
# EEG Recording - Patient ID: 12345
# Date: 2024-09-25
# Sampling Rate: 300 Hz
Time,Fz,Cz,Pz,X1:LEOG
0.000,12.5,8.3,15.2,250.3
0.003,13.1,8.7,15.8,251.2
```

## Tips for Best Results

1. **Large Files**: For files with >1M samples, consider:
   - Processing in chunks
   - Downsampling for overview visualization
   - Focusing on specific time windows

2. **Channel Names**: The tool recognizes:
   - Standard 10-20 EEG positions (Fz, Cz, etc.)
   - ECG channels with "EOG" or "ECG" in the name
   - Reference channels named "CM" or "REF"

3. **Browser Performance**: For best interactivity:
   - Use Chrome or Firefox
   - Close other browser tabs when viewing large datasets
   - Consider time windowing for very long recordings

## Troubleshooting

### Issue: Channels not displayed correctly
- Check channel naming matches expected conventions
- Verify CSV format (comma-separated, no extra whitespace)
- Ensure Time column is present

### Issue: HTML file doesn't open
- Verify file path is correct
- Try opening with different browser
- Check file wasn't corrupted during generation

### Issue: Plot appears empty
- Confirm data file contains actual data (not just headers)
- Check for proper numeric format (not strings)
- Verify no NaN or infinite values in critical columns

## Contributing Examples

Have an interesting use case? Please contribute!

1. Add your example code/data to this directory
2. Update this README with description
3. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
