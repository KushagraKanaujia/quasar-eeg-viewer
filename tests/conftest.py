"""
Pytest configuration and fixtures for tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_eeg_data():
    """Create sample EEG/ECG data for testing"""
    np.random.seed(42)
    n_samples = 1000
    time = np.linspace(0, 10, n_samples)

    data = {
        'Time': time,
        'Fz': np.random.randn(n_samples) * 10 + 50,
        'Cz': np.random.randn(n_samples) * 10 + 45,
        'P3': np.random.randn(n_samples) * 10 + 40,
        'X1:LEOG': np.random.randn(n_samples) * 100 + 500,
        'X2:REOG': np.random.randn(n_samples) * 100 + 480,
        'CM': np.random.randn(n_samples) * 50 + 1200,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_eeg_data, tmp_path):
    """Create a temporary CSV file with sample data"""
    csv_file = tmp_path / "test_data.csv"
    sample_eeg_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_csv_with_comments(sample_eeg_data, tmp_path):
    """Create a CSV file with comment lines"""
    csv_file = tmp_path / "test_data_comments.csv"

    with open(csv_file, 'w') as f:
        f.write("# This is a comment\n")
        f.write("# Another comment line\n")
        sample_eeg_data.to_csv(f, index=False)

    return csv_file


@pytest.fixture
def empty_csv_file(tmp_path):
    """Create an empty CSV file"""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    return csv_file


@pytest.fixture
def invalid_csv_file(tmp_path):
    """Create an invalid CSV file"""
    csv_file = tmp_path / "invalid.csv"
    csv_file.write_text("This is not valid CSV\ndata")
    return csv_file
