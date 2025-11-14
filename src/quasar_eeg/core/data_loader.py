"""
Data loading module for EEG/ECG signals

Handles reading and parsing CSV files containing physiological signal data.
Supports comment filtering, validation, and error handling for various CSV formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import io

from ..utils.logger import Logger
from .config import ApplicationConfig, DEFAULT_CONFIG


class DataLoadError(Exception):
    """
    Raised when something goes wrong loading EEG/ECG data files.

    This usually happens when:
    - File doesn't exist or can't be read
    - CSV is malformed or empty
    - Required columns are missing (like 'Time')
    - File is corrupted or not actually a CSV
    """
    pass


class EEGDataLoader:
    """Loader for EEG/ECG CSV data with robust error handling"""

    def __init__(self, config: Optional[ApplicationConfig] = None):
        """
        Initialize data loader

        Args:
            config: Application configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = Logger.get_logger(__name__, self.config.log_level)

    def load_from_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load EEG/ECG data from CSV file

        This is the main entry point for loading signal data. It handles:
        - File validation (exists, not too big, etc.)
        - Comment line filtering (lines starting with #)
        - CSV parsing with pandas
        - Data validation (has Time column, has signal channels, etc.)

        Args:
            filepath: Path to CSV file containing signal data

        Returns:
            DataFrame with loaded data, validated and ready for classification

        Raises:
            DataLoadError: If file cannot be loaded or parsed
        """
        filepath = Path(filepath)

        # First, make sure the file actually exists
        if not filepath.exists():
            error_msg = f"Data file not found: {filepath}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

        # Check file size - warn if it's huge (might take a while)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            self.logger.warning(
                f"Large file detected: {file_size_mb:.1f}MB. "
                f"This may take some time to process."
            )

        self.logger.info(f"Loading data from: {filepath}")

        try:
            # Read the whole file into memory
            # (could optimize this for huge files, but works fine for typical EEG recordings)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Filter out comment lines (lines starting with #)
            # Also skip completely empty lines to avoid pandas errors
            data_lines = [
                line for line in lines
                if not line.strip().startswith('#') and line.strip()
            ]

            if not data_lines:
                raise DataLoadError("No data found in file (only comments or empty)")

            # Now parse it as CSV using pandas
            # The io.StringIO trick lets us pass the filtered lines to pandas
            data_string = ''.join(data_lines)
            df = pd.read_csv(io.StringIO(data_string))

            # Make sure the data looks valid (has Time column, etc.)
            self._validate_dataframe(df)

            # Log some useful info about what we loaded
            duration = df['Time'].max() if 'Time' in df.columns else 0
            self.logger.info(
                f"Loaded {len(df)} samples, {len(df.columns)} channels. "
                f"Duration: {duration:.1f}s"
            )

            return df

        except pd.errors.EmptyDataError:
            error_msg = f"Empty or invalid CSV file: {filepath}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

        except pd.errors.ParserError as e:
            error_msg = f"CSV parsing error: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error loading data: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate loaded DataFrame has required structure

        Args:
            df: DataFrame to validate

        Raises:
            DataLoadError: If validation fails
        """
        # Check for Time column
        if 'Time' not in df.columns:
            raise DataLoadError("CSV must contain 'Time' column")

        # Check for at least one signal channel
        signal_columns = [
            col for col in df.columns
            if col not in self.config.channel_config.ignore_columns
        ]

        if not signal_columns:
            raise DataLoadError("No signal channels found in CSV")

        # Check for NaN values in Time column
        if df['Time'].isna().any():
            self.logger.warning("NaN values found in Time column, will be handled")

        # Check time monotonicity
        if not df['Time'].is_monotonic_increasing:
            self.logger.warning("Time column is not monotonically increasing")

        self.logger.debug(f"DataFrame validation passed. Channels: {signal_columns}")

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load from existing DataFrame (for programmatic use)

        Args:
            df: DataFrame with signal data

        Returns:
            Validated DataFrame

        Raises:
            DataLoadError: If validation fails
        """
        self._validate_dataframe(df)
        return df.copy()

    def get_file_info(self, filepath: Union[str, Path]) -> dict:
        """
        Get metadata about a data file without loading it entirely

        Args:
            filepath: Path to CSV file

        Returns:
            Dictionary with file metadata
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {'error': 'File not found'}

        file_size = filepath.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # Try to read just the header
        try:
            with open(filepath, 'r') as f:
                # Skip comment lines
                for line in f:
                    if not line.strip().startswith('#'):
                        header = line.strip().split(',')
                        break

            return {
                'filename': filepath.name,
                'size_bytes': file_size,
                'size_mb': file_size_mb,
                'columns': header,
                'num_columns': len(header)
            }

        except Exception as e:
            return {'error': str(e)}
