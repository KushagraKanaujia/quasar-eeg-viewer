"""
Configuration management for Quasar EEG Viewer

Centralized configuration for EEG/ECG signal processing and visualization.
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ChannelConfig:
    """Configuration for channel classification"""

    # Standard 10-20 system EEG electrodes
    eeg_channels: List[str] = field(default_factory=lambda: [
        'Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4',
        'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2',
        'F7', 'F8', 'A1', 'A2', 'Pz'
    ])

    # ECG/EOG channels
    ecg_channels: List[str] = field(default_factory=lambda: [
        'X1:LEOG', 'X2:REOG'
    ])

    # Reference channels
    reference_channels: List[str] = field(default_factory=lambda: ['CM'])

    # Columns to ignore during processing
    ignore_columns: List[str] = field(default_factory=lambda: [
        'Time', 'Trigger', 'Time_Offset', 'ADC_Status',
        'ADC_Sequence', 'Event', 'Comments', 'X3:'
    ])


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""

    # Default plot dimensions
    plot_height: int = 700
    plot_width: int = 1200

    # Default time window (seconds)
    default_time_window: int = 30

    # Color palettes
    eeg_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

    ecg_colors: List[str] = field(default_factory=lambda: [
        '#d62728', '#ff7f0e'
    ])

    reference_color: str = '#7f7f7f'

    # Export settings
    export_format: str = 'png'
    export_scale: int = 2

    # Sampling rate (Hz)
    default_sampling_rate: int = 300

    # Unit conversion factors
    uv_to_mv_factor: float = 1000.0


@dataclass
class ApplicationConfig:
    """Main application configuration"""

    channel_config: ChannelConfig = field(default_factory=ChannelConfig)
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Logging
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Data processing
    max_file_size_mb: int = 500
    chunk_size: int = 10000

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ApplicationConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'channel_config': self.channel_config.__dict__,
            'visualization_config': self.visualization_config.__dict__,
            'log_level': self.log_level,
            'log_format': self.log_format,
            'max_file_size_mb': self.max_file_size_mb,
            'chunk_size': self.chunk_size,
        }


# Global default configuration
DEFAULT_CONFIG = ApplicationConfig()
