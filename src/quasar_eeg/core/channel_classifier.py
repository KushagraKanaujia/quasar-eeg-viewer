"""
Channel classification module for EEG/ECG signals

Automatically categorizes signal channels based on naming conventions.
This is super useful because EEG and ECG signals have vastly different amplitudes,
so we need to know which is which to scale them properly on the plot.
"""

import pandas as pd
from typing import Dict, List
from ..utils.logger import Logger
from .config import ApplicationConfig, DEFAULT_CONFIG


class ChannelClassifier:
    """
    Classifier for EEG, ECG, and reference channels

    This does the heavy lifting of figuring out what type each channel is.
    It looks at the column names and matches them against known patterns:
    - EEG electrodes follow the 10-20 system (Fz, Cz, P3, etc.)
    - ECG/EOG channels usually have "EOG" or "ECG" in the name
    - Reference channels are things like CM (common mode)

    If it doesn't recognize a channel, it gets marked as "unknown" so you can
    decide what to do with it.
    """

    def __init__(self, config: ApplicationConfig = None):
        """
        Initialize channel classifier

        Args:
            config: Application configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = Logger.get_logger(__name__, self.config.log_level)

    def classify(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Classify channels into EEG, ECG, and reference groups

        This is where the magic happens - we look at each column name and figure out
        what type of signal it is. This matters because EEG signals are measured in
        microvolts (tiny!) while ECG is in millivolts (bigger), so they need different
        y-axes on the plot.

        Args:
            df: DataFrame containing signal data

        Returns:
            Dictionary with channel classifications:
                - 'eeg': List of EEG channel names (brain electrodes)
                - 'ecg': List of ECG/EOG channel names (heart, eye movement)
                - 'reference': List of reference channel names
                - 'ignored': List of non-signal columns (Time, triggers, etc.)
                - 'unknown': List of unclassified channels (will warn about these)
        """
        channel_config = self.config.channel_config

        # Get all the column names from the dataframe
        available_columns = df.columns.tolist()

        # Match column names against our known EEG electrode positions
        eeg_channels = [
            col for col in available_columns
            if col in channel_config.eeg_channels
        ]

        # Same for ECG/EOG channels
        ecg_channels = [
            col for col in available_columns
            if col in channel_config.ecg_channels
        ]

        # And reference channels
        reference_channels = [
            col for col in available_columns
            if col in channel_config.reference_channels
        ]

        # Figure out if there are any channels we don't recognize
        # (useful for catching typos or non-standard naming)
        all_known = (
            set(eeg_channels) |
            set(ecg_channels) |
            set(reference_channels) |
            set(channel_config.ignore_columns)
        )

        unknown_channels = [
            col for col in available_columns
            if col not in all_known
        ]

        classifications = {
            'eeg': eeg_channels,
            'ecg': ecg_channels,
            'reference': reference_channels,
            'ignored': channel_config.ignore_columns,
            'unknown': unknown_channels
        }

        # Log what we found so users can see what's being plotted
        self.logger.info(f"Channel classification:")
        self.logger.info(f"  EEG channels: {len(eeg_channels)} - {eeg_channels}")
        self.logger.info(f"  ECG/EOG channels: {len(ecg_channels)} - {ecg_channels}")
        self.logger.info(f"  Reference channels: {len(reference_channels)} - {reference_channels}")

        # Warn about unknown channels - might be a typo or non-standard naming
        if unknown_channels:
            self.logger.warning(f"  Unknown channels: {len(unknown_channels)} - {unknown_channels}")

        return classifications

    def get_signal_channels(self, df: pd.DataFrame) -> List[str]:
        """
        Get all signal channels (EEG + ECG + Reference)

        Args:
            df: DataFrame containing signal data

        Returns:
            List of signal channel names
        """
        classifications = self.classify(df)
        return (
            classifications['eeg'] +
            classifications['ecg'] +
            classifications['reference']
        )

    def add_custom_channel(
        self,
        channel_name: str,
        channel_type: str
    ) -> None:
        """
        Add a custom channel to classification

        Args:
            channel_name: Name of the channel to add
            channel_type: Type ('eeg', 'ecg', 'reference', or 'ignore')

        Raises:
            ValueError: If channel_type is invalid
        """
        valid_types = ['eeg', 'ecg', 'reference', 'ignore']
        if channel_type not in valid_types:
            raise ValueError(
                f"Invalid channel type: {channel_type}. "
                f"Must be one of {valid_types}"
            )

        channel_config = self.config.channel_config

        if channel_type == 'eeg':
            if channel_name not in channel_config.eeg_channels:
                channel_config.eeg_channels.append(channel_name)
                self.logger.info(f"Added EEG channel: {channel_name}")

        elif channel_type == 'ecg':
            if channel_name not in channel_config.ecg_channels:
                channel_config.ecg_channels.append(channel_name)
                self.logger.info(f"Added ECG channel: {channel_name}")

        elif channel_type == 'reference':
            if channel_name not in channel_config.reference_channels:
                channel_config.reference_channels.append(channel_name)
                self.logger.info(f"Added reference channel: {channel_name}")

        elif channel_type == 'ignore':
            if channel_name not in channel_config.ignore_columns:
                channel_config.ignore_columns.append(channel_name)
                self.logger.info(f"Added ignored column: {channel_name}")

    def validate_classifications(
        self,
        classifications: Dict[str, List[str]]
    ) -> bool:
        """
        Validate that classifications contain at least one signal channel

        Args:
            classifications: Channel classifications dictionary

        Returns:
            True if valid, False otherwise
        """
        total_signals = (
            len(classifications.get('eeg', [])) +
            len(classifications.get('ecg', [])) +
            len(classifications.get('reference', []))
        )

        if total_signals == 0:
            self.logger.error("No signal channels found after classification")
            return False

        return True
