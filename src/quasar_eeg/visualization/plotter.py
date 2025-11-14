"""
Visualization module for EEG/ECG signals

Creates interactive Plotly visualizations with dual-axis scaling.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..utils.logger import Logger
from ..core.config import ApplicationConfig, DEFAULT_CONFIG


class PlotterError(Exception):
    """Custom exception for plotting errors"""
    pass


class EEGPlotter:
    """Interactive plotter for multi-channel EEG/ECG signals"""

    def __init__(self, config: Optional[ApplicationConfig] = None):
        """
        Initialize plotter

        Args:
            config: Application configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = Logger.get_logger(__name__, self.config.log_level)
        self.viz_config = self.config.visualization_config

    def create_plot(
        self,
        df: pd.DataFrame,
        channels: Dict[str, List[str]],
        output_file: Union[str, Path] = 'eeg_ecg_plot.html',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive Plotly plot with dual y-axes

        Args:
            df: DataFrame with signal data
            channels: Channel classifications from ChannelClassifier
            output_file: Output HTML filename
            title: Optional custom plot title

        Returns:
            Plotly Figure object

        Raises:
            PlotterError: If plotting fails
        """
        try:
            self.logger.info("Creating interactive visualization...")

            # Validate inputs
            self._validate_inputs(df, channels)

            # Create figure with dual y-axes
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=(title or "EEG/ECG Multichannel Visualization",)
            )

            # Add traces for each channel type
            self._add_eeg_traces(fig, df, channels['eeg'])
            self._add_ecg_traces(fig, df, channels['ecg'])
            self._add_reference_traces(fig, df, channels['reference'])

            # Configure layout
            self._configure_layout(fig, df)

            # Save to file
            self._save_plot(fig, output_file)

            # Log summary
            total_channels = (
                len(channels['eeg']) +
                len(channels['ecg']) +
                len(channels['reference'])
            )
            self.logger.info(f"Plot created successfully with {total_channels} channels")

            return fig

        except Exception as e:
            error_msg = f"Failed to create plot: {str(e)}"
            self.logger.error(error_msg)
            raise PlotterError(error_msg)

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        channels: Dict[str, List[str]]
    ) -> None:
        """Validate inputs for plotting"""
        if df.empty:
            raise PlotterError("DataFrame is empty")

        if 'Time' not in df.columns:
            raise PlotterError("DataFrame must contain 'Time' column")

        total_channels = (
            len(channels.get('eeg', [])) +
            len(channels.get('ecg', [])) +
            len(channels.get('reference', []))
        )

        if total_channels == 0:
            raise PlotterError("No channels to plot")

    def _add_eeg_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        eeg_channels: List[str]
    ) -> None:
        """Add EEG channel traces to figure"""
        for i, channel in enumerate(eeg_channels):
            if channel not in df.columns:
                self.logger.warning(f"EEG channel {channel} not found in data")
                continue

            color = self.viz_config.eeg_colors[i % len(self.viz_config.eeg_colors)]

            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=df[channel],
                    mode='lines',
                    name=f'{channel} (µV)',
                    line=dict(color=color, width=1),
                    yaxis='y',
                    legendgroup='eeg',
                    legendgrouptitle_text="EEG Channels (µV)"
                ),
                secondary_y=False
            )

        self.logger.debug(f"Added {len(eeg_channels)} EEG traces")

    def _add_ecg_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        ecg_channels: List[str]
    ) -> None:
        """Add ECG/EOG channel traces to figure"""
        for i, channel in enumerate(ecg_channels):
            if channel not in df.columns:
                self.logger.warning(f"ECG channel {channel} not found in data")
                continue

            # Convert µV to mV for better visualization
            ecg_data_mv = df[channel] / self.viz_config.uv_to_mv_factor
            color = self.viz_config.ecg_colors[i % len(self.viz_config.ecg_colors)]

            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=ecg_data_mv,
                    mode='lines',
                    name=f'{channel} (mV)',
                    line=dict(color=color, width=2, dash='dash'),
                    yaxis='y2',
                    legendgroup='ecg',
                    legendgrouptitle_text="ECG/EOG Channels (mV)"
                ),
                secondary_y=True
            )

        self.logger.debug(f"Added {len(ecg_channels)} ECG traces")

    def _add_reference_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        reference_channels: List[str]
    ) -> None:
        """Add reference channel traces to figure"""
        for channel in reference_channels:
            if channel not in df.columns:
                self.logger.warning(f"Reference channel {channel} not found in data")
                continue

            # Convert µV to mV for display
            ref_data_mv = df[channel] / self.viz_config.uv_to_mv_factor

            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=ref_data_mv,
                    mode='lines',
                    name=f'{channel} (Reference)',
                    line=dict(
                        color=self.viz_config.reference_color,
                        width=1,
                        dash='dot'
                    ),
                    yaxis='y2',
                    opacity=0.7,
                    legendgroup='reference',
                    legendgrouptitle_text="Reference"
                ),
                secondary_y=True
            )

        self.logger.debug(f"Added {len(reference_channels)} reference traces")

    def _configure_layout(self, fig: go.Figure, df: pd.DataFrame) -> None:
        """Configure plot layout and styling"""
        duration = df['Time'].max()
        num_samples = len(df)

        # Calculate sampling rate
        if len(df) > 1:
            time_diff = df['Time'].iloc[1] - df['Time'].iloc[0]
            sampling_rate = 1.0 / time_diff if time_diff > 0 else self.viz_config.default_sampling_rate
        else:
            sampling_rate = self.viz_config.default_sampling_rate

        fig.update_layout(
            title={
                'text': (
                    f'EEG/ECG Signal Visualization<br>'
                    f'<sub>Duration: {duration:.1f}s | '
                    f'Sampling Rate: {sampling_rate:.0f} Hz | '
                    f'{num_samples} samples</sub>'
                ),
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Time (seconds)',
            height=self.viz_config.plot_height,
            hovermode='x unified',

            # Enable range slider for navigation
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05),
                type='linear',
                range=[0, min(self.viz_config.default_time_window, duration)]
            ),

            # Interaction settings
            dragmode='pan',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01,
                groupclick="toggleitem"
            )
        )

        # Configure y-axes
        fig.update_yaxes(
            title_text="EEG Amplitude (µV)",
            secondary_y=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )

        fig.update_yaxes(
            title_text="ECG/Reference Amplitude (mV)",
            secondary_y=True,
            showgrid=False
        )

    def _save_plot(self, fig: go.Figure, output_file: Union[str, Path]) -> None:
        """Save plot to HTML file"""
        output_path = Path(output_file)

        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': [
                'drawline', 'drawopenpath', 'drawclosedpath',
                'drawcircle', 'drawrect', 'eraseshape'
            ],
            'toImageButtonOptions': {
                'format': self.viz_config.export_format,
                'filename': output_path.stem,
                'height': self.viz_config.plot_height,
                'width': self.viz_config.plot_width,
                'scale': self.viz_config.export_scale
            }
        }

        fig.write_html(
            str(output_path),
            include_plotlyjs=True,
            config=config
        )

        self.logger.info(f"Plot saved to: {output_path}")
        self.logger.info(f"Open {output_path} in your browser to explore the data")

    def create_subplot_by_channel(
        self,
        df: pd.DataFrame,
        channels: Dict[str, List[str]],
        output_file: Union[str, Path] = 'eeg_ecg_subplots.html'
    ) -> go.Figure:
        """
        Create subplot layout with each channel in its own row

        Args:
            df: DataFrame with signal data
            channels: Channel classifications
            output_file: Output HTML filename

        Returns:
            Plotly Figure object
        """
        all_channels = channels['eeg'] + channels['ecg'] + channels['reference']
        num_channels = len(all_channels)

        if num_channels == 0:
            raise PlotterError("No channels to plot")

        fig = make_subplots(
            rows=num_channels,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=all_channels
        )

        for idx, channel in enumerate(all_channels, start=1):
            if channel in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Time'],
                        y=df[channel],
                        mode='lines',
                        name=channel,
                        showlegend=False
                    ),
                    row=idx,
                    col=1
                )

        fig.update_layout(
            height=200 * num_channels,
            title_text="EEG/ECG Channels (Subplot View)"
        )

        fig.write_html(str(output_file))
        self.logger.info(f"Subplot view saved to: {output_file}")

        return fig
