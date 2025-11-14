"""
Tests for EEGDataLoader
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quasar_eeg.core.data_loader import EEGDataLoader, DataLoadError
from quasar_eeg.core.config import ApplicationConfig


class TestEEGDataLoader:
    """Test suite for EEGDataLoader"""

    def test_load_valid_csv(self, sample_csv_file):
        """Test loading a valid CSV file"""
        loader = EEGDataLoader()
        df = loader.load_from_file(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Time' in df.columns
        assert len(df) == 1000

    def test_load_csv_with_comments(self, sample_csv_with_comments):
        """Test loading CSV with comment lines"""
        loader = EEGDataLoader()
        df = loader.load_from_file(sample_csv_with_comments)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Time' in df.columns

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist"""
        loader = EEGDataLoader()

        with pytest.raises(DataLoadError, match="not found"):
            loader.load_from_file("nonexistent_file.csv")

    def test_load_empty_file(self, empty_csv_file):
        """Test loading an empty CSV file"""
        loader = EEGDataLoader()

        with pytest.raises(DataLoadError):
            loader.load_from_file(empty_csv_file)

    def test_load_from_dataframe(self, sample_eeg_data):
        """Test loading from existing DataFrame"""
        loader = EEGDataLoader()
        df = loader.load_from_dataframe(sample_eeg_data)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Time' in df.columns

    def test_validate_missing_time_column(self, sample_eeg_data):
        """Test validation fails when Time column is missing"""
        loader = EEGDataLoader()
        df_no_time = sample_eeg_data.drop('Time', axis=1)

        with pytest.raises(DataLoadError, match="Time"):
            loader._validate_dataframe(df_no_time)

    def test_get_file_info(self, sample_csv_file):
        """Test getting file metadata"""
        loader = EEGDataLoader()
        info = loader.get_file_info(sample_csv_file)

        assert 'filename' in info
        assert 'size_mb' in info
        assert 'columns' in info
        assert info['filename'] == 'test_data.csv'

    def test_get_file_info_nonexistent(self):
        """Test file info for nonexistent file"""
        loader = EEGDataLoader()
        info = loader.get_file_info("nonexistent.csv")

        assert 'error' in info

    def test_custom_config(self, sample_csv_file):
        """Test loader with custom configuration"""
        config = ApplicationConfig()
        config.log_level = 'DEBUG'
        config.max_file_size_mb = 100

        loader = EEGDataLoader(config)
        df = loader.load_from_file(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
