"""
Quasar EEG Viewer - Interactive visualization tool for EEG/ECG signals

A professional, scalable tool for visualizing multi-channel electrophysiological data.
"""

__version__ = '2.0.0'
__author__ = 'Kushagra Kanaujia'
__license__ = 'MIT'

from .core.data_loader import EEGDataLoader, DataLoadError
from .core.channel_classifier import ChannelClassifier
from .core.config import (
    ApplicationConfig,
    ChannelConfig,
    VisualizationConfig,
    DEFAULT_CONFIG
)
from .visualization.plotter import EEGPlotter, PlotterError
from .utils.logger import Logger

__all__ = [
    'EEGDataLoader',
    'DataLoadError',
    'ChannelClassifier',
    'EEGPlotter',
    'PlotterError',
    'ApplicationConfig',
    'ChannelConfig',
    'VisualizationConfig',
    'DEFAULT_CONFIG',
    'Logger',
    '__version__',
]


def quick_plot(
    filepath: str,
    output_file: str = 'eeg_ecg_plot.html',
    config=None
):
    """
    Quick plotting function for convenience

    Args:
        filepath: Path to CSV data file
        output_file: Output HTML filename
        config: Optional configuration

    Returns:
        Plotly Figure object

    Example:
        >>> from quasar_eeg import quick_plot
        >>> fig = quick_plot('data.csv', 'output.html')
    """
    config = config or DEFAULT_CONFIG

    # Load data
    loader = EEGDataLoader(config)
    df = loader.load_from_file(filepath)

    # Classify channels
    classifier = ChannelClassifier(config)
    channels = classifier.classify(df)

    # Create plot
    plotter = EEGPlotter(config)
    fig = plotter.create_plot(df, channels, output_file)

    return fig
