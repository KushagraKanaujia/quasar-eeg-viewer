#!/usr/bin/env python3
"""
QUASAR EEG/ECG Multichannel Plotter
Interactive visualization of EEG and ECG signals with dual-axis scaling
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os


def load_quasar_data(filepath):
    """
    Load QUASAR CSV data, skipping comment lines starting with #

    Returns:
        pandas.DataFrame: Loaded data with proper channel classification
    """
    print(f"Loading data from {filepath}")

    # Read file and skip comment lines
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter out comment lines starting with #
    data_lines = [line for line in lines if not line.strip().startswith('#')]

    # Parse CSV data
    import io
    data_string = ''.join(data_lines)
    df = pd.read_csv(io.StringIO(data_string))

    print(f"Loaded {len(df)} samples, {len(df.columns)} channels")
    print(f"Duration: {df['Time'].max():.1f} seconds")

    return df


def classify_channels(df):
    """
    Classify channels according to QUASAR specification

    Returns:
        dict: Channel classifications
    """
    # Ignore columns as per QUASAR spec
    ignore_columns = ['Time', 'Trigger', 'Time_Offset', 'ADC_Status',
                     'ADC_Sequence', 'Event', 'Comments', 'X3:']

    # EEG channels (µV): Standard 10-20 system electrodes
    eeg_channels = ['Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4',
                   'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2',
                   'F7', 'F8', 'A1', 'A2', 'Pz']

    # ECG channels (mV): Eye movement/ECG channels
    ecg_channels = ['X1:LEOG', 'X2:REOG']  # Left and Right ECG

    # Reference channel
    reference_channels = ['CM']  # Common Mode reference

    # Find actual channels in data
    available_channels = [col for col in df.columns if col not in ignore_columns]

    classifications = {
        'eeg': [ch for ch in available_channels if ch in eeg_channels],
        'ecg': [ch for ch in available_channels if ch in ecg_channels],
        'reference': [ch for ch in available_channels if ch in reference_channels],
        'ignored': ignore_columns
    }

    print(f"EEG channels ({len(classifications['eeg'])}): {classifications['eeg']}")
    print(f"ECG channels ({len(classifications['ecg'])}): {classifications['ecg']}")
    print(f"Reference channels ({len(classifications['reference'])}): {classifications['reference']}")

    return classifications


def create_interactive_plot(df, channels, output_file='eeg_ecg_plot.html'):
    """
    Create interactive Plotly plot with dual y-axes for EEG/ECG scaling

    Args:
        df: DataFrame with signal data
        channels: Channel classifications
        output_file: Output HTML filename
    """
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=("QUASAR EEG/ECG Multichannel Plot",)
    )

    # Color palettes
    eeg_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    ecg_colors = ['#d62728', '#ff7f0e']  # Red/orange for ECG
    ref_color = '#7f7f7f'  # Gray for reference

    # Add EEG traces (µV scale - primary y-axis)
    for i, channel in enumerate(channels['eeg']):
        if channel in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=df[channel],  # Data already in µV
                    mode='lines',
                    name=f'{channel} (µV)',
                    line=dict(color=eeg_colors[i % len(eeg_colors)], width=1),
                    yaxis='y',
                    legendgroup='eeg',
                    legendgrouptitle_text="EEG Channels (µV)"
                ),
                secondary_y=False
            )

    # Add ECG traces (mV scale - secondary y-axis)
    for i, channel in enumerate(channels['ecg']):
        if channel in df.columns:
            # Convert µV to mV for ECG display (QUASAR ECG channels are high amplitude)
            ecg_data_mv = df[channel] / 1000  # Convert µV to mV
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=ecg_data_mv,
                    mode='lines',
                    name=f'{channel} (mV)',
                    line=dict(color=ecg_colors[i % len(ecg_colors)], width=2, dash='dash'),
                    yaxis='y2',
                    legendgroup='ecg',
                    legendgrouptitle_text="ECG Channels (mV)"
                ),
                secondary_y=True
            )

    # Add reference traces (mV scale - secondary y-axis)
    for channel in channels['reference']:
        if channel in df.columns:
            ref_data_mv = df[channel] / 1000  # Convert µV to mV for display
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=ref_data_mv,
                    mode='lines',
                    name=f'{channel} (Reference)',
                    line=dict(color=ref_color, width=1, dash='dot'),
                    yaxis='y2',
                    opacity=0.7,
                    legendgroup='reference',
                    legendgrouptitle_text="Reference"
                ),
                secondary_y=True
            )

    # Update layout for optimal scrolling/zooming
    fig.update_layout(
        title={
            'text': f'QUASAR EEG/ECG Data Visualization<br><sub>Duration: {df["Time"].max():.1f}s | Sampling Rate: 300 Hz | {len(df)} samples</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Time (seconds)',
        height=700,
        hovermode='x unified',

        # Enable range slider for scrolling
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type='linear',
            range=[0, min(30, df['Time'].max())]  # Show first 30 seconds by default
        ),

        # Optimize for interaction
        dragmode='pan',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            groupclick="toggleitem"  # Allow individual trace toggling
        )
    )

    # Set y-axes labels and properties
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
        showgrid=False  # Avoid grid overlap
    )

    # Save as standalone HTML file
    fig.write_html(
        output_file,
        include_plotlyjs=True,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath',
                                   'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'eeg_ecg_plot',
                'height': 700,
                'width': 1200,
                'scale': 2
            }
        }
    )

    print(f"Interactive plot saved as: {output_file}")
    print(f"Open {output_file} in your web browser to explore the data")

    return fig


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='QUASAR EEG/ECG Interactive Plotter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eeg_ecg_plotter.py                                    # Use default data file
  python eeg_ecg_plotter.py --data "EEG and ECG data_02_raw.csv"
  python eeg_ecg_plotter.py --output my_plot.html
        """
    )

    parser.add_argument(
        '--data', '-d',
        default='EEG and ECG data_02_raw.csv',
        help='Path to QUASAR CSV data file (default: EEG and ECG data_02_raw.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        default='eeg_ecg_plot.html',
        help='Output HTML filename (default: eeg_ecg_plot.html)'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found")
        print("Please ensure the QUASAR CSV file is in the current directory")
        return 1

    try:
        # Load and process data
        df = load_quasar_data(args.data)
        channels = classify_channels(df)

        # Create interactive plot
        fig = create_interactive_plot(df, channels, args.output)

        print("\n" + "="*60)
        print("SUCCESS: Interactive EEG/ECG plot generated!")
        print("="*60)
        print(f"📊 Total channels plotted: {len(channels['eeg']) + len(channels['ecg']) + len(channels['reference'])}")
        print(f"🧠 EEG channels (µV): {len(channels['eeg'])}")
        print(f"💓 ECG channels (mV): {len(channels['ecg'])}")
        print(f"📡 Reference channels: {len(channels['reference'])}")
        print(f"⏱️  Duration: {df['Time'].max():.1f} seconds")
        print(f"📈 Sampling rate: 300 Hz")
        print(f"📁 Output file: {args.output}")
        print("\nFeatures:")
        print("• Scroll/pan/zoom with mouse or range slider")
        print("• Toggle channels on/off via legend")
        print("• Dual y-axis scaling (EEG in µV, ECG in mV)")
        print("• Export plot as PNG via toolbar")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())