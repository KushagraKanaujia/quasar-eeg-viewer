"""
Tests for EEGPlotter
"""

import pytest
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quasar_eeg.visualization.plotter import EEGPlotter, PlotterError
from quasar_eeg.core.channel_classifier import ChannelClassifier


class TestEEGPlotter:
    """Test suite for EEGPlotter"""

    def test_create_plot_basic(self, sample_eeg_data, tmp_path):
        """Test basic plot creation"""
        classifier = ChannelClassifier()
        channels = classifier.classify(sample_eeg_data)

        plotter = EEGPlotter()
        output_file = tmp_path / "test_plot.html"

        fig = plotter.create_plot(sample_eeg_data, channels, output_file)

        assert isinstance(fig, go.Figure)
        assert output_file.exists()

    def test_create_plot_empty_dataframe(self):
        """Test plot creation fails with empty DataFrame"""
        plotter = EEGPlotter()
        empty_df = pd.DataFrame()
        channels = {'eeg': [], 'ecg': [], 'reference': []}

        with pytest.raises(PlotterError, match="empty"):
            plotter.create_plot(empty_df, channels, "output.html")

    def test_create_plot_no_time_column(self, sample_eeg_data):
        """Test plot creation fails without Time column"""
        plotter = EEGPlotter()
        df_no_time = sample_eeg_data.drop('Time', axis=1)
        channels = {'eeg': ['Fz'], 'ecg': [], 'reference': []}

        with pytest.raises(PlotterError, match="Time"):
            plotter.create_plot(df_no_time, channels, "output.html")

    def test_create_plot_no_channels(self, sample_eeg_data):
        """Test plot creation fails with no channels"""
        plotter = EEGPlotter()
        channels = {'eeg': [], 'ecg': [], 'reference': []}

        with pytest.raises(PlotterError, match="No channels"):
            plotter.create_plot(sample_eeg_data, channels, "output.html")

    def test_create_subplot_by_channel(self, sample_eeg_data, tmp_path):
        """Test subplot creation"""
        classifier = ChannelClassifier()
        channels = classifier.classify(sample_eeg_data)

        plotter = EEGPlotter()
        output_file = tmp_path / "test_subplot.html"

        fig = plotter.create_subplot_by_channel(
            sample_eeg_data, channels, output_file
        )

        assert isinstance(fig, go.Figure)
        assert output_file.exists()

    def test_add_eeg_traces(self, sample_eeg_data):
        """Test adding EEG traces to figure"""
        from plotly.subplots import make_subplots

        plotter = EEGPlotter()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        eeg_channels = ['Fz', 'Cz', 'P3']
        plotter._add_eeg_traces(fig, sample_eeg_data, eeg_channels)

        assert len(fig.data) == 3

    def test_add_ecg_traces(self, sample_eeg_data):
        """Test adding ECG traces to figure"""
        from plotly.subplots import make_subplots

        plotter = EEGPlotter()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        ecg_channels = ['X1:LEOG', 'X2:REOG']
        plotter._add_ecg_traces(fig, sample_eeg_data, ecg_channels)

        assert len(fig.data) == 2

    def test_add_reference_traces(self, sample_eeg_data):
        """Test adding reference traces to figure"""
        from plotly.subplots import make_subplots

        plotter = EEGPlotter()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        reference_channels = ['CM']
        plotter._add_reference_traces(fig, sample_eeg_data, reference_channels)

        assert len(fig.data) == 1

    def test_configure_layout(self, sample_eeg_data):
        """Test plot layout configuration"""
        from plotly.subplots import make_subplots

        plotter = EEGPlotter()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        plotter._configure_layout(fig, sample_eeg_data)

        assert fig.layout.xaxis.title.text == 'Time (seconds)'
        assert fig.layout.height == 700

    def test_save_plot(self, sample_eeg_data, tmp_path):
        """Test saving plot to file"""
        from plotly.subplots import make_subplots

        plotter = EEGPlotter()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        output_file = tmp_path / "saved_plot.html"

        plotter._save_plot(fig, output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0
