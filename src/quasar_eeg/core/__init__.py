"""Core modules for data loading and processing"""

from .data_loader import EEGDataLoader, DataLoadError
from .channel_classifier import ChannelClassifier
from .config import (
    ApplicationConfig,
    ChannelConfig,
    VisualizationConfig,
    DEFAULT_CONFIG
)

__all__ = [
    'EEGDataLoader',
    'DataLoadError',
    'ChannelClassifier',
    'ApplicationConfig',
    'ChannelConfig',
    'VisualizationConfig',
    'DEFAULT_CONFIG',
]
