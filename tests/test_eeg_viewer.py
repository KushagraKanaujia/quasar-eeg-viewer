#!/usr/bin/env python3
"""
Unit tests for QUASAR EEG Viewer
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from eeg_viewer import EEGViewer


class TestEEGViewer(unittest.TestCase):
    """Test cases for the EEG Viewer application."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        self.sample_data = self.create_sample_data()
        self.temp_file = None

    def tearDown(self):
        """Clean up after each test method."""
        if self.temp_file and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def create_sample_data(self):
        """Create sample EEG/ECG data for testing."""
        duration = 10  # seconds
        sampling_rate = 1000  # Hz
        t = np.linspace(0, duration, duration * sampling_rate, False)

        data = {
            'Time': t,
            'EEG_C3': 10e-6 * np.sin(2 * np.pi * 10 * t) + 2e-6 * np.random.normal(0, 1, len(t)),
            'EEG_C4': 12e-6 * np.sin(2 * np.pi * 8 * t) + 2e-6 * np.random.normal(0, 1, len(t)),
            'ECG_I': 1e-3 * np.sin(2 * np.pi * 1.2 * t) + 0.1e-3 * np.random.normal(0, 1, len(t)),
            'ECG_II': 1.2e-3 * np.sin(2 * np.pi * 1.2 * t) + 0.1e-3 * np.random.normal(0, 1, len(t)),
        }

        return pd.DataFrame(data)

    def create_temp_csv(self, data):
        """Create a temporary CSV file with the given data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f, index=False)
            self.temp_file = f.name
        return self.temp_file

    def test_initialization(self):
        """Test EEGViewer initialization."""
        viewer = EEGViewer()
        self.assertIsNotNone(viewer.app)
        self.assertIsNone(viewer.data)
        self.assertIsInstance(viewer.config, dict)
        self.assertEqual(viewer.channels, [])

    def test_config_loading(self):
        """Test configuration loading with defaults."""
        viewer = EEGViewer()

        # Check default configuration values
        self.assertEqual(viewer.config['sampling_rate'], 1000)
        self.assertEqual(viewer.config['eeg_scale_factor'], 1e-6)
        self.assertEqual(viewer.config['ecg_scale_factor'], 1e-3)
        self.assertIn('eeg_channels', viewer.config)
        self.assertIn('ecg_channels', viewer.config)

    def test_data_loading(self):
        """Test data loading from CSV file."""
        csv_file = self.create_temp_csv(self.sample_data)
        viewer = EEGViewer()

        # Load data
        viewer.load_data(csv_file)

        # Verify data was loaded correctly
        self.assertIsNotNone(viewer.data)
        self.assertEqual(len(viewer.data), len(self.sample_data))
        self.assertIn('Time', viewer.data.columns)

        # Check channel detection
        self.assertEqual(len(viewer.eeg_channels), 2)  # EEG_C3, EEG_C4
        self.assertEqual(len(viewer.ecg_channels), 2)  # ECG_I, ECG_II
        self.assertIn('EEG_C3', viewer.eeg_channels)
        self.assertIn('ECG_I', viewer.ecg_channels)

    def test_channel_classification(self):
        """Test automatic channel classification."""
        csv_file = self.create_temp_csv(self.sample_data)
        viewer = EEGViewer()
        viewer.load_data(csv_file)

        # Test EEG channel detection
        for channel in viewer.eeg_channels:
            self.assertTrue('EEG' in channel.upper() or any(
                eeg in channel.upper() for eeg in ['C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4']
            ))

        # Test ECG channel detection
        for channel in viewer.ecg_channels:
            self.assertTrue('ECG' in channel.upper())

    def test_data_with_missing_time_column(self):
        """Test data loading when time column is missing."""
        data_no_time = self.sample_data.drop('Time', axis=1)
        csv_file = self.create_temp_csv(data_no_time)
        viewer = EEGViewer()
        viewer.load_data(csv_file)

        # Should create a Time column automatically
        self.assertIn('Time', viewer.data.columns)
        self.assertEqual(len(viewer.data), len(data_no_time))

        # Time should be based on sampling rate
        expected_time = np.arange(len(data_no_time)) / viewer.sampling_rate
        np.testing.assert_array_almost_equal(viewer.data['Time'].values, expected_time)

    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        empty_data = pd.DataFrame()
        csv_file = self.create_temp_csv(empty_data)
        viewer = EEGViewer()

        with self.assertRaises(Exception):
            viewer.load_data(csv_file)

    def test_data_statistics(self):
        """Test data statistics calculation."""
        csv_file = self.create_temp_csv(self.sample_data)
        viewer = EEGViewer()
        viewer.load_data(csv_file)

        # Check that channels have reasonable statistics
        for channel in viewer.eeg_channels + viewer.ecg_channels:
            data_column = viewer.data[channel]
            self.assertIsNotNone(data_column.mean())
            self.assertIsNotNone(data_column.std())
            self.assertGreater(data_column.std(), 0)  # Should have some variation

    def test_data_integrity(self):
        """Test data integrity after loading."""
        csv_file = self.create_temp_csv(self.sample_data)
        viewer = EEGViewer()
        viewer.load_data(csv_file)

        # Check that data shape is maintained
        self.assertEqual(len(viewer.data), len(self.sample_data))

        # Check that original channels are preserved
        original_channels = set(self.sample_data.columns) - {'Time'}
        loaded_channels = set(viewer.channels)
        self.assertEqual(original_channels, loaded_channels)

        # Check that time column is monotonic
        time_diff = np.diff(viewer.data['Time'])
        self.assertTrue(np.all(time_diff >= 0))  # Should be non-decreasing

    def test_large_data_handling(self):
        """Test handling of larger datasets."""
        # Create larger dataset
        duration = 60  # seconds
        sampling_rate = 1000
        t = np.linspace(0, duration, duration * sampling_rate, False)

        large_data = {
            'Time': t,
            'EEG_C3': 10e-6 * np.sin(2 * np.pi * 10 * t),
            'EEG_C4': 12e-6 * np.sin(2 * np.pi * 8 * t),
            'ECG_I': 1e-3 * np.sin(2 * np.pi * 1.2 * t),
        }

        large_df = pd.DataFrame(large_data)
        csv_file = self.create_temp_csv(large_df)
        viewer = EEGViewer()
        viewer.load_data(csv_file)

        # Should handle large data without issues
        self.assertEqual(len(viewer.data), len(large_df))
        self.assertGreater(viewer.data['Time'].max(), 50)  # Should be at least 50 seconds

    def test_mixed_channel_types(self):
        """Test data with mixed channel naming conventions."""
        mixed_data = pd.DataFrame({
            'Time': np.linspace(0, 10, 10000),
            'C3': 10e-6 * np.random.normal(0, 1, 10000),  # EEG without prefix
            'eeg_o1': 12e-6 * np.random.normal(0, 1, 10000),  # Lowercase EEG
            'ECG_LEAD_I': 1e-3 * np.random.normal(0, 1, 10000),  # ECG with LEAD
            'Unknown_Signal': 5e-6 * np.random.normal(0, 1, 10000),  # Unknown type
        })

        csv_file = self.create_temp_csv(mixed_data)
        viewer = EEGViewer()
        viewer.load_data(csv_file)

        # Should classify channels appropriately
        self.assertGreater(len(viewer.eeg_channels), 0)
        self.assertGreater(len(viewer.ecg_channels), 0)


class TestDataGeneration(unittest.TestCase):
    """Test cases for sample data generation."""

    def test_sample_data_structure(self):
        """Test that generated sample data has correct structure."""
        # This would test the generate_sample_data.py script
        # For now, just ensure we can import it
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
            import generate_sample_data
            self.assertTrue(hasattr(generate_sample_data, 'generate_sample_dataset'))
        except ImportError:
            self.fail("Could not import generate_sample_data module")


if __name__ == '__main__':
    # Set up test environment
    unittest.main(verbosity=2)