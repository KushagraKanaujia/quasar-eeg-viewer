#!/usr/bin/env python3
"""
EEG/ECG Multichannel Viewer - Legacy Entry Point

This script is maintained for backward compatibility.
For new projects, please use the modular package:
    from quasar_eeg import quick_plot
    quick_plot('data.csv', 'output.html')

Or use the CLI directly:
    python -m quasar_eeg.cli --data data.csv --output output.html

Author: Kushagra Kanaujia
Repository: https://github.com/KushagraKanaujia/quasar-eeg-viewer
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from quasar_eeg.cli import main


# Legacy function wrappers for backward compatibility
def load_eeg_data(filepath):
    """
    Legacy wrapper for loading EEG data

    Note: This function is deprecated. Use quasar_eeg.core.EEGDataLoader instead.
    """
    from quasar_eeg.core import EEGDataLoader
    loader = EEGDataLoader()
    return loader.load_from_file(filepath)


def classify_channels(df):
    """
    Legacy wrapper for channel classification

    Note: This function is deprecated. Use quasar_eeg.core.ChannelClassifier instead.
    """
    from quasar_eeg.core import ChannelClassifier
    classifier = ChannelClassifier()
    return classifier.classify(df)


def create_interactive_plot(df, channels, output_file='eeg_ecg_plot.html'):
    """
    Legacy wrapper for creating plots

    Note: This function is deprecated. Use quasar_eeg.visualization.EEGPlotter instead.
    """
    from quasar_eeg.visualization import EEGPlotter
    plotter = EEGPlotter()
    return plotter.create_plot(df, channels, output_file)


if __name__ == '__main__':
    exit(main())