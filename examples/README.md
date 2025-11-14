# Usage Examples

This directory contains example scripts showing how to use Quasar EEG Viewer.

## Basic Usage (`basic_usage.py`)

Shows five different ways to use the tool:

1. **Quick plot** - one-liner for simple use cases
2. **Full control** - load, classify, and plot separately
3. **Custom config** - set logging levels and other options
4. **Custom electrodes** - add non-standard electrode positions
5. **Subplot view** - each channel in its own row

## Running the Examples

```bash
# Make sure you're in the quasar-eeg-viewer directory
cd quasar-eeg-viewer

# Install the package first
pip install -e .

# Run the examples (you'll need to provide your own CSV file)
python examples/basic_usage.py
```

## Example Data Format

Your CSV should look something like this:

```csv
Time,Fz,Cz,Pz,X1:LEOG,X2:REOG,CM
0.000,12.5,8.3,15.2,250.3,245.1,1200.5
0.003,13.1,8.7,15.8,251.2,246.0,1205.3
0.007,11.9,9.1,14.9,249.8,244.5,1198.2
...
```

Key points:
- First column must be `Time` (in seconds)
- Other columns are your signal channels
- Use standard 10-20 electrode names for EEG (Fz, Cz, Pz, etc.)
- EOG channels typically named like X1:LEOG, X2:REOG
- Lines starting with # are treated as comments

## Need Help?

Check the main [README](../README.md) for more detailed documentation, or open an issue on GitHub if you're stuck.
