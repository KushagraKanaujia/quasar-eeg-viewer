# API Reference

Complete API documentation for the EEG/ECG Multichannel Viewer.

## Module: `eeg_ecg_plotter`

The main module containing all data processing and visualization functions.

---

## Functions

### `load_eeg_data(filepath)`

Load EEG/ECG signal data from a CSV file with automatic comment line filtering.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `filepath` | `str` | Absolute or relative path to the CSV file containing signal data |

#### Returns

| Type | Description |
|------|-------------|
| `pandas.DataFrame` | DataFrame with time series data, where each column represents a channel and each row is a time sample |

#### Raises

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If the specified file doesn't exist |
| `pd.errors.ParserError` | If CSV parsing fails |
| `ValueError` | If file is empty or improperly formatted |

#### Behavior

- Automatically skips lines starting with `#` (comment lines)
- Preserves all numeric columns
- Maintains original data types
- Prints loading progress to stdout

#### Example

```python
from eeg_ecg_plotter import load_eeg_data

# Load data
df = load_eeg_data("recordings/patient_001.csv")

# Check what was loaded
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Duration: {df['Time'].max():.2f} seconds")
print(f"Samples: {len(df)}")
```

#### Example Output

```
Loading data from recordings/patient_001.csv
Loaded 90000 samples, 25 channels
Duration: 300.0 seconds
```

---

### `classify_channels(df)`

Automatically classify channels into EEG, ECG/EOG, and reference groups based on naming conventions.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `df` | `pandas.DataFrame` | DataFrame containing signal data with column names representing channels |

#### Returns

| Type | Description |
|------|-------------|
| `dict` | Dictionary with four keys:<br>- `'eeg'`: List of EEG channel names<br>- `'ecg'`: List of ECG/EOG channel names<br>- `'reference'`: List of reference channel names<br>- `'ignored'`: List of non-signal columns |

#### Classification Rules

**EEG Channels** (Standard 10-20 system positions):
- Frontal: `Fz`, `F3`, `F4`, `F7`, `F8`, `Fp1`, `Fp2`
- Central: `Cz`, `C3`, `C4`
- Temporal: `T3`, `T4`, `T5`, `T6`
- Parietal: `Pz`, `P3`, `P4`
- Occipital: `O1`, `O2`
- Auricular: `A1`, `A2`

**ECG/EOG Channels**:
- Channels containing "EOG" or "ECG" in the name
- Examples: `X1:LEOG`, `X2:REOG`, `ECG1`

**Reference Channels**:
- `CM` (Common Mode reference)
- `REF`

**Ignored Columns**:
- `Time`, `Trigger`, `Time_Offset`, `ADC_Status`, `ADC_Sequence`, `Event`, `Comments`, `X3:`

#### Example

```python
from eeg_ecg_plotter import load_eeg_data, classify_channels

df = load_eeg_data("data.csv")
channels = classify_channels(df)

print(f"EEG channels ({len(channels['eeg'])}): {channels['eeg']}")
print(f"ECG channels ({len(channels['ecg'])}): {channels['ecg']}")
print(f"Reference channels ({len(channels['reference'])}): {channels['reference']}")
```

#### Example Output

```
EEG channels (21): ['Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4', 'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'A1', 'A2', 'Pz']
ECG channels (2): ['X1:LEOG', 'X2:REOG']
Reference channels (1): ['CM']
```

#### Custom Classification

You can override automatic classification by manually creating the dictionary:

```python
custom_channels = {
    'eeg': ['Fz', 'Cz', 'Pz'],  # Only midline channels
    'ecg': ['ECG1'],
    'reference': [],
    'ignored': ['Time', 'Trigger']
}
```

---

### `create_interactive_plot(df, channels, output_file='eeg_ecg_plot.html')`

Generate an interactive Plotly visualization with dual y-axes for EEG and ECG signals.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pandas.DataFrame` | *required* | DataFrame containing signal data with 'Time' column |
| `channels` | `dict` | *required* | Channel classifications from `classify_channels()` |
| `output_file` | `str` | `'eeg_ecg_plot.html'` | Output filename for the HTML visualization |

#### Returns

| Type | Description |
|------|-------------|
| `plotly.graph_objects.Figure` | Plotly Figure object that can be further customized |

#### Raises

| Exception | Condition |
|-----------|-----------|
| `KeyError` | If 'Time' column is missing from DataFrame |
| `ValueError` | If DataFrame is empty or channels dict is invalid |

#### Visualization Features

**Layout:**
- Dual y-axis configuration (EEG on left, ECG/Reference on right)
- Grouped legend with channel type categories
- Built-in range slider for time navigation
- Unified hover showing values across all channels

**EEG Traces (Primary Y-axis):**
- Units: µV (microvolts)
- Solid lines with color-coded palette
- Legend group: "EEG Channels (µV)"

**ECG Traces (Secondary Y-axis):**
- Units: mV (converted from µV by dividing by 1000)
- Dashed lines in red/orange tones
- Legend group: "ECG Channels (mV)"

**Reference Traces (Secondary Y-axis):**
- Units: mV (converted from µV)
- Dotted gray line with 70% opacity
- Legend group: "Reference"

**Interactive Controls:**
- Pan: Drag mode enabled by default
- Zoom: Scroll wheel or box select
- Range slider: Continuous scrolling through time
- Channel toggle: Click legend items
- Export: Camera icon exports to PNG

#### Configuration

The function applies these Plotly configurations:

```python
{
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath',
                            'drawcircle', 'drawrect', 'eraseshape'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'eeg_ecg_plot',
        'height': 700,
        'width': 1200,
        'scale': 2
    }
}
```

#### Example

```python
from eeg_ecg_plotter import load_eeg_data, classify_channels, create_interactive_plot

# Load and process
df = load_eeg_data("data.csv")
channels = classify_channels(df)

# Create visualization
fig = create_interactive_plot(df, channels, "my_visualization.html")

# Further customize if needed
fig.update_layout(title="Custom Title")
fig.write_html("custom_output.html")
```

#### Example Output (Console)

```
Interactive plot saved as: my_visualization.html
Open my_visualization.html in your web browser to explore the data
```

#### Advanced Customization

The returned Figure object can be further customized:

```python
fig = create_interactive_plot(df, channels)

# Modify layout
fig.update_layout(
    height=900,
    width=1600,
    title="Custom EEG Analysis",
    font=dict(size=14)
)

# Modify traces
fig.update_traces(
    line=dict(width=0.5),  # Thinner lines
    selector=dict(legendgroup='eeg')
)

# Save with custom name
fig.write_html("custom_plot.html")
```

---

### `main()`

Command-line interface entry point. Parses arguments and orchestrates the visualization pipeline.

#### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--data` | `-d` | `str` | `'EEG and ECG data_02_raw.csv'` | Path to input CSV file |
| `--output` | `-o` | `str` | `'eeg_ecg_plot.html'` | Output HTML filename |
| `--help` | `-h` | - | - | Show help message |

#### Returns

| Type | Description |
|------|-------------|
| `int` | Exit code: 0 for success, 1 for error |

#### Example Usage

```bash
# Default behavior
python eeg_ecg_plotter.py

# Custom arguments
python eeg_ecg_plotter.py --data "recordings/session_01.csv" --output "session_01_plot.html"

# Short form
python eeg_ecg_plotter.py -d data.csv -o output.html
```

---

## Data Structures

### Channel Classification Dictionary

Structure returned by `classify_channels()`:

```python
{
    'eeg': [                    # List of EEG channel names
        'Fz', 'Cz', 'Pz', ...
    ],
    'ecg': [                    # List of ECG/EOG channel names
        'X1:LEOG', 'X2:REOG'
    ],
    'reference': [              # List of reference channel names
        'CM'
    ],
    'ignored': [                # List of non-signal columns
        'Time', 'Trigger', 'Time_Offset', ...
    ]
}
```

---

## Constants

### Color Palettes

**EEG Color Palette:**
```python
['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
```

**ECG Color Palette:**
```python
['#d62728', '#ff7f0e']  # Red and orange
```

**Reference Color:**
```python
'#7f7f7f'  # Gray
```

---

## Usage Patterns

### Basic Pipeline

```python
from eeg_ecg_plotter import load_eeg_data, classify_channels, create_interactive_plot

# Step 1: Load data
df = load_eeg_data("input.csv")

# Step 2: Classify channels
channels = classify_channels(df)

# Step 3: Create visualization
fig = create_interactive_plot(df, channels, "output.html")
```

### Batch Processing

```python
import os
from pathlib import Path

def process_directory(input_dir, output_dir):
    """Process all CSV files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for csv_file in input_path.glob("*.csv"):
        try:
            df = load_eeg_data(str(csv_file))
            channels = classify_channels(df)

            output_file = output_path / f"{csv_file.stem}_plot.html"
            create_interactive_plot(df, channels, str(output_file))

            print(f"✓ Processed: {csv_file.name}")
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")

# Usage
process_directory("data/recordings", "output/visualizations")
```

### Custom Time Window

```python
def plot_time_window(filepath, start_time, end_time, output_file):
    """Create visualization for a specific time window."""
    df = load_eeg_data(filepath)

    # Filter to time window
    df_window = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)].copy()

    # Reset time to start at 0
    df_window['Time'] = df_window['Time'] - start_time

    channels = classify_channels(df_window)
    fig = create_interactive_plot(df_window, channels, output_file)

    return fig

# Usage
plot_time_window("data.csv", start_time=30, end_time=60, output_file="30_60s.html")
```

---

## Dependencies

### Required Libraries

| Library | Minimum Version | Purpose |
|---------|-----------------|---------|
| `pandas` | 1.5.0 | Data manipulation and CSV parsing |
| `plotly` | 5.15.0 | Interactive visualization |
| `numpy` | 1.21.0 | Numerical operations |

### Standard Library Dependencies

- `argparse`: Command-line argument parsing
- `os`: File system operations
- `io`: String I/O for CSV processing

---

## Error Handling

### Common Exceptions

```python
try:
    df = load_eeg_data("data.csv")
    channels = classify_channels(df)
    create_interactive_plot(df, channels, "output.html")
except FileNotFoundError:
    print("Error: Data file not found")
except pd.errors.ParserError:
    print("Error: Invalid CSV format")
except KeyError as e:
    print(f"Error: Required column missing: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Performance Considerations

### Memory Usage

For large datasets:
- DataFrame memory: ~8 bytes per float value
- Example: 300 Hz × 300 seconds × 24 channels = ~17 MB
- HTML output: 2-3× the raw data size

### Processing Time

Typical performance on modern hardware:
- Data loading: ~1-2 seconds per 100MB
- Channel classification: <0.1 seconds
- Plot generation: 2-5 seconds for 90,000 samples
- HTML writing: 1-2 seconds

### Optimization Tips

1. **Downsampling** for large files:
```python
df = load_eeg_data("large_file.csv")
df_downsampled = df[::5]  # Keep every 5th sample
```

2. **Chunk processing**:
```python
chunk_size = 90000  # 5 minutes at 300 Hz
for i in range(0, len(df), chunk_size):
    chunk = df[i:i+chunk_size]
    create_interactive_plot(chunk, channels, f"chunk_{i//chunk_size}.html")
```

3. **Selective channels**:
```python
# Keep only channels of interest
channels_to_plot = ['Time', 'Fz', 'Cz', 'Pz', 'O1', 'O2']
df_subset = df[channels_to_plot]
```

---

## Version History

### Version 1.0.0
- Initial release
- Basic EEG/ECG visualization
- Dual y-axis scaling
- Interactive pan/zoom/scroll
- Channel toggling
- PNG export
- Command-line interface
- Python API

---

## Future API Extensions

Planned additions for future versions:

- `filter_signal(df, lowcut, highcut)`: Bandpass filtering
- `detect_artifacts(df, channels)`: Automatic artifact detection
- `compute_spectrogram(df, channel)`: Frequency analysis
- `export_selection(df, start, end, filename)`: Export time windows
- `compare_recordings(df1, df2)`: Multi-file visualization

---

For practical examples, see [USAGE.md](USAGE.md) and [examples/README.md](../examples/README.md).
