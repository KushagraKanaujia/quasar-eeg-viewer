"""
Tests for ChannelClassifier
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quasar_eeg.core.channel_classifier import ChannelClassifier
from quasar_eeg.core.config import ApplicationConfig


class TestChannelClassifier:
    """Test suite for ChannelClassifier"""

    def test_classify_standard_channels(self, sample_eeg_data):
        """Test classification of standard EEG/ECG channels"""
        classifier = ChannelClassifier()
        channels = classifier.classify(sample_eeg_data)

        assert 'eeg' in channels
        assert 'ecg' in channels
        assert 'reference' in channels
        assert 'ignored' in channels
        assert 'unknown' in channels

        # Check EEG channels
        assert 'Fz' in channels['eeg']
        assert 'Cz' in channels['eeg']
        assert 'P3' in channels['eeg']

        # Check ECG channels
        assert 'X1:LEOG' in channels['ecg']
        assert 'X2:REOG' in channels['ecg']

        # Check reference channels
        assert 'CM' in channels['reference']

    def test_classify_unknown_channels(self):
        """Test classification with unknown channels"""
        df = pd.DataFrame({
            'Time': [0, 1, 2],
            'Fz': [1, 2, 3],
            'UnknownChannel': [4, 5, 6]
        })

        classifier = ChannelClassifier()
        channels = classifier.classify(df)

        assert 'UnknownChannel' in channels['unknown']

    def test_get_signal_channels(self, sample_eeg_data):
        """Test getting all signal channels"""
        classifier = ChannelClassifier()
        signal_channels = classifier.get_signal_channels(sample_eeg_data)

        assert isinstance(signal_channels, list)
        assert len(signal_channels) > 0
        assert 'Fz' in signal_channels
        assert 'X1:LEOG' in signal_channels
        assert 'CM' in signal_channels
        assert 'Time' not in signal_channels

    def test_add_custom_eeg_channel(self, sample_eeg_data):
        """Test adding custom EEG channel"""
        classifier = ChannelClassifier()
        classifier.add_custom_channel('CustomEEG', 'eeg')

        # Add custom channel to dataframe
        df = sample_eeg_data.copy()
        df['CustomEEG'] = [1] * len(df)

        channels = classifier.classify(df)
        assert 'CustomEEG' in channels['eeg']

    def test_add_custom_ecg_channel(self, sample_eeg_data):
        """Test adding custom ECG channel"""
        classifier = ChannelClassifier()
        classifier.add_custom_channel('CustomECG', 'ecg')

        df = sample_eeg_data.copy()
        df['CustomECG'] = [1] * len(df)

        channels = classifier.classify(df)
        assert 'CustomECG' in channels['ecg']

    def test_add_custom_invalid_type(self):
        """Test adding custom channel with invalid type"""
        classifier = ChannelClassifier()

        with pytest.raises(ValueError, match="Invalid channel type"):
            classifier.add_custom_channel('CustomChannel', 'invalid_type')

    def test_validate_classifications_success(self, sample_eeg_data):
        """Test validation succeeds with signal channels"""
        classifier = ChannelClassifier()
        channels = classifier.classify(sample_eeg_data)

        assert classifier.validate_classifications(channels) is True

    def test_validate_classifications_failure(self):
        """Test validation fails with no signal channels"""
        classifier = ChannelClassifier()
        channels = {
            'eeg': [],
            'ecg': [],
            'reference': []
        }

        assert classifier.validate_classifications(channels) is False

    def test_custom_config(self, sample_eeg_data):
        """Test classifier with custom configuration"""
        config = ApplicationConfig()
        config.log_level = 'DEBUG'

        classifier = ChannelClassifier(config)
        channels = classifier.classify(sample_eeg_data)

        assert isinstance(channels, dict)
        assert 'eeg' in channels
