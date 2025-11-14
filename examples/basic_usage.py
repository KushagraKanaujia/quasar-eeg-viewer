#!/usr/bin/env python3
"""
Basic usage examples for Quasar EEG Viewer

This shows a few different ways to use the tool in your own code.
"""

# Example 1: Super simple - one line
# =====================================
from quasar_eeg import quick_plot

# Just give it a CSV file and output filename
fig = quick_plot('sample_data.csv', 'output.html')
print("Created output.html - open it in your browser!")


# Example 2: More control over the process
# =========================================
from quasar_eeg import EEGDataLoader, ChannelClassifier, EEGPlotter

# Load the data
loader = EEGDataLoader()
df = loader.load_from_file('sample_data.csv')

# Figure out which channels are EEG vs ECG
classifier = ChannelClassifier()
channels = classifier.classify(df)

# Make the plot
plotter = EEGPlotter()
fig = plotter.create_plot(df, channels, 'custom_output.html')

# Now you have a Plotly figure object you can modify further if needed
print(f"Loaded {len(df)} samples from {len(channels['eeg'])} EEG channels")


# Example 3: Custom configuration
# ================================
from quasar_eeg import ApplicationConfig, EEGDataLoader, ChannelClassifier, EEGPlotter

# Set up custom config (maybe you want debug logging)
config = ApplicationConfig()
config.log_level = 'DEBUG'

# Pass the config to each component
loader = EEGDataLoader(config)
df = loader.load_from_file('sample_data.csv')

classifier = ChannelClassifier(config)
channels = classifier.classify(df)

plotter = EEGPlotter(config)
fig = plotter.create_plot(df, channels, 'debug_output.html')


# Example 4: Adding custom electrode positions
# =============================================
from quasar_eeg import ChannelClassifier

classifier = ChannelClassifier()

# Got a custom electrode that's not in the standard 10-20 system?
# Add it like this:
classifier.add_custom_channel('MyCustomElectrode', 'eeg')

# Now it'll be recognized when you classify channels


# Example 5: Create a subplot view
# =================================
from quasar_eeg import EEGDataLoader, ChannelClassifier, EEGPlotter

loader = EEGDataLoader()
df = loader.load_from_file('sample_data.csv')

classifier = ChannelClassifier()
channels = classifier.classify(df)

plotter = EEGPlotter()
# Each channel gets its own row - good for detailed inspection
fig = plotter.create_subplot_by_channel(df, channels, 'subplot_output.html')


print("\nAll examples completed!")
print("Check the generated HTML files in your browser.")
