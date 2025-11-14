"""
Command-line interface for Quasar EEG Viewer

Provides user-friendly CLI for visualizing EEG/ECG data.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .core.data_loader import EEGDataLoader, DataLoadError
from .core.channel_classifier import ChannelClassifier
from .core.config import ApplicationConfig, DEFAULT_CONFIG
from .visualization.plotter import EEGPlotter, PlotterError
from .utils.logger import Logger


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Quasar EEG Viewer - Interactive visualization for EEG/ECG signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default data file
  quasar-eeg-viewer

  # Specify custom data file
  quasar-eeg-viewer --data path/to/data.csv

  # Custom output file
  quasar-eeg-viewer --data data.csv --output analysis.html

  # Verbose logging
  quasar-eeg-viewer --data data.csv --log-level DEBUG

  # Create subplot view
  quasar-eeg-viewer --data data.csv --subplot

For more information, visit: https://github.com/KushagraKanaujia/quasar-eeg-viewer
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'Quasar EEG Viewer v{__version__}'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        default='EEG and ECG data_02_raw.csv',
        help='Path to CSV data file (default: EEG and ECG data_02_raw.csv)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='eeg_ecg_plot.html',
        help='Output HTML filename (default: eeg_ecg_plot.html)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--subplot',
        action='store_true',
        help='Create subplot view with each channel in separate row'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Display file information without plotting'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (JSON)'
    )

    return parser


def load_custom_config(config_path: str) -> ApplicationConfig:
    """Load configuration from JSON file"""
    import json

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return ApplicationConfig.from_dict(config_dict)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        print("Using default configuration")
        return DEFAULT_CONFIG


def display_file_info(filepath: str, config: ApplicationConfig) -> int:
    """Display information about data file"""
    loader = EEGDataLoader(config)
    info = loader.get_file_info(filepath)

    if 'error' in info:
        print(f"Error: {info['error']}")
        return 1

    print("=" * 60)
    print("FILE INFORMATION")
    print("=" * 60)
    print(f"Filename: {info['filename']}")
    print(f"Size: {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
    print(f"Columns: {info['num_columns']}")
    print(f"Channel names: {', '.join(info['columns'])}")
    print("=" * 60)

    return 0


def run_visualization(args: argparse.Namespace, config: ApplicationConfig) -> int:
    """Run the main visualization pipeline"""
    logger = Logger.get_logger(__name__, args.log_level)

    try:
        # Validate input file
        input_path = Path(args.data)
        if not input_path.exists():
            logger.error(f"Data file not found: {input_path}")
            logger.error("Please ensure the CSV file exists and the path is correct")
            return 1

        # Load data
        logger.info("Starting EEG/ECG visualization pipeline...")
        loader = EEGDataLoader(config)
        df = loader.load_from_file(args.data)

        # Classify channels
        classifier = ChannelClassifier(config)
        channels = classifier.classify(df)

        # Validate classifications
        if not classifier.validate_classifications(channels):
            logger.error("Channel classification failed: No signal channels found")
            return 1

        # Create plot
        plotter = EEGPlotter(config)

        if args.subplot:
            logger.info("Creating subplot visualization...")
            fig = plotter.create_subplot_by_channel(df, channels, args.output)
        else:
            logger.info("Creating interactive visualization...")
            fig = plotter.create_plot(df, channels, args.output)

        # Display summary
        print("\n" + "=" * 60)
        print("SUCCESS: Interactive EEG/ECG plot generated!")
        print("=" * 60)
        print(f"Total channels plotted: {len(channels['eeg']) + len(channels['ecg']) + len(channels['reference'])}")
        print(f"  EEG channels (µV): {len(channels['eeg'])}")
        print(f"  ECG/EOG channels (mV): {len(channels['ecg'])}")
        print(f"  Reference channels: {len(channels['reference'])}")
        print(f"Duration: {df['Time'].max():.1f} seconds")
        print(f"Total samples: {len(df):,}")
        print(f"Output file: {args.output}")
        print("\nFeatures:")
        print("  • Pan, zoom, and scroll through data")
        print("  • Toggle channels on/off via legend")
        print("  • Dual y-axis scaling (EEG in µV, ECG in mV)")
        print("  • Export plot as PNG via toolbar")
        print("  • Interactive hover with unified crosshair")
        print("=" * 60)

        return 0

    except DataLoadError as e:
        logger.error(f"Data loading error: {e}")
        return 1

    except PlotterError as e:
        logger.error(f"Plotting error: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI

    Args:
        argv: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load configuration
    if args.config:
        config = load_custom_config(args.config)
    else:
        config = DEFAULT_CONFIG

    # Set log level
    config.log_level = args.log_level

    # Handle info mode
    if args.info:
        return display_file_info(args.data, config)

    # Run visualization
    return run_visualization(args, config)


if __name__ == '__main__':
    sys.exit(main())
