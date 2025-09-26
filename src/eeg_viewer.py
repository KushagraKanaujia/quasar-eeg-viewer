#!/usr/bin/env python3
"""
QUASAR EEG/ECG Data Visualization Tool

A professional interactive visualization application for EEG and ECG signals
with dual-axis scaling, channel toggles, and export capabilities.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EEGViewer:
    """Main EEG/ECG visualization application class."""

    def __init__(self, data_file=None, config_file=None):
        """Initialize the EEG viewer application."""
        self.app = dash.Dash(__name__)
        self.data = None
        self.config = self._load_config(config_file)
        self.channels = []
        self.eeg_channels = []
        self.ecg_channels = []
        self.cm_channels = []
        self.sampling_rate = self.config.get('sampling_rate', 300)  # Hz - QUASAR default

        if data_file:
            self.load_data(data_file)

        self._setup_layout()
        self._setup_callbacks()

    def _load_config(self, config_file):
        """Load configuration from JSON file."""
        default_config = {
            'sampling_rate': 300,  # QUASAR default sampling rate
            'data_units': 'uV',  # QUASAR data is already in µV
            'eeg_scale_factor': 1.0,  # No conversion needed - data already in µV
            'ecg_scale_factor': 1e-3,  # Convert µV to mV for ECG display
            'window_size': 10.0,  # seconds
            'eeg_channels': ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz'],
            'ecg_channels': ['X1:LEOG', 'X2:REOG'],
            'colors': {
                'eeg': '#1f77b4',
                'ecg': '#d62728',
                'background': '#ffffff',
                'grid': '#f0f0f0'
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")

        return default_config

    def _validate_file_path(self, data_file):
        """Validate file path and accessibility."""
        if not data_file:
            raise ValueError("No data file path provided")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        if not os.access(data_file, os.R_OK):
            raise PermissionError(f"Cannot read data file: {data_file}")

        # Check file size (warn if > 100MB)
        file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
        if file_size > 100:
            logger.warning(f"Large file detected: {file_size:.1f}MB. Loading may take time.")

    def _validate_csv_structure(self, df):
        """Validate CSV data structure and content."""
        if df.empty:
            raise ValueError("CSV file is empty")

        if len(df.columns) < 2:
            raise ValueError("CSV must have at least 2 columns (time + signal)")

        # Check for numeric data in signal columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 1:
            raise ValueError("No numeric columns found in CSV")

        # Check for missing data
        if df.isnull().sum().sum() > len(df) * 0.1:  # More than 10% missing
            logger.warning("Significant missing data detected (>10% of values)")

        logger.info(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")

    def load_data(self, data_file):
        """Load EEG/ECG data from CSV file with QUASAR format support."""
        try:
            logger.info(f"Loading data from {data_file}")

            # Validate file before processing
            self._validate_file_path(data_file)

            # Read file handling # comment lines and extracting metadata
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract metadata from QUASAR header comments
            metadata = {}
            for line in lines:
                if line.startswith('#'):
                    if 'Sample_Frequency_(Hz)' in line:
                        try:
                            freq_value = line.split('=')[1].split(',')[1].strip()
                            freq_float = float(freq_value)
                            if freq_float <= 0:
                                raise ValueError("Invalid sampling rate")
                            metadata['sampling_rate'] = freq_float
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Could not parse sampling rate: {e}")
                    elif 'Sensor_Data_Units' in line:
                        try:
                            units = line.split('=')[1].split(',')[1].strip()
                            metadata['data_units'] = units
                        except IndexError:
                            logger.warning("Could not parse data units")

            # Update sampling rate if found in file
            if 'sampling_rate' in metadata:
                self.sampling_rate = metadata['sampling_rate']
                logger.info(f"Detected sampling rate: {self.sampling_rate} Hz")

            # Filter out comment lines starting with #
            data_lines = [line for line in lines if not line.strip().startswith('#')]

            if not data_lines:
                raise ValueError("No data rows found after removing comments")

            # Write filtered data to temporary string for pandas
            import io
            data_string = ''.join(data_lines)

            try:
                self.data = pd.read_csv(io.StringIO(data_string))
            except pd.errors.EmptyDataError:
                raise ValueError("CSV file contains no data")
            except pd.errors.ParserError as e:
                raise ValueError(f"CSV parsing error: {e}")

            # Validate the loaded data
            self._validate_csv_structure(self.data)

            # Detect channels automatically - exclude non-signal columns
            ignore_columns = ['Time', 'time', 'timestamp', 'Trigger', 'Time_Offset', 'ADC_Status',
                            'ADC_Sequence', 'Event', 'Comments', 'X3:']
            self.channels = [col for col in self.data.columns
                           if not any(ignore in col for ignore in ignore_columns)]

            # QUASAR-specific channel classification
            # EEG channels: Fz, Cz, P3, C3, F3, F4, C4, P4, Fp1, Fp2, T3, T4, T5, T6, O1, O2, F7, F8, A1, A2, Pz
            eeg_names = ['Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4', 'Fp1', 'Fp2',
                        'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'A1', 'A2', 'Pz']

            # ECG channels: X1:LEOG (Left ECG), X2:REOG (Right ECG)
            ecg_names = ['X1:LEOG', 'X2:REOG']

            self.eeg_channels = [ch for ch in self.channels if ch in eeg_names]
            self.ecg_channels = [ch for ch in self.channels if ch in ecg_names]

            # CM (Common Mode reference) - treat separately
            self.cm_channels = [ch for ch in self.channels if ch == 'CM']

            # Handle remaining channels (fallback detection)
            remaining = [ch for ch in self.channels
                        if ch not in self.eeg_channels + self.ecg_channels + self.cm_channels]

            # Try fallback detection for remaining channels
            for ch in remaining:
                if any(eeg in ch.upper() for eeg in ['EEG', 'C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4']):
                    self.eeg_channels.append(ch)
                elif any(ecg in ch.upper() for ecg in ['ECG', 'EOG', 'LEAD']):
                    self.ecg_channels.append(ch)

            # Create time axis if not present
            if 'Time' not in self.data.columns and 'time' not in self.data.columns:
                self.data['Time'] = np.arange(len(self.data)) / self.sampling_rate
            elif 'time' in self.data.columns:
                self.data['Time'] = self.data['time']

            logger.info(f"Loaded {len(self.data)} samples with {len(self.channels)} channels")
            logger.info(f"EEG channels ({len(self.eeg_channels)}): {self.eeg_channels}")
            logger.info(f"ECG channels ({len(self.ecg_channels)}): {self.ecg_channels}")
            logger.info(f"CM channels ({len(self.cm_channels)}): {self.cm_channels}")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _setup_layout(self):
        """Setup the Dash application layout."""
        self.app.layout = html.Div([
            # Header section with improved styling
            html.Div([
                html.H1("QUASAR EEG/ECG Signal Viewer",
                       style={
                           'textAlign': 'center',
                           'marginBottom': '10px',
                           'color': '#2c3e50',
                           'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
                           'fontWeight': '300',
                           'fontSize': '2.5em'
                       }),
                html.P("Professional Signal Visualization and Analysis Tool",
                       style={
                           'textAlign': 'center',
                           'color': '#7f8c8d',
                           'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
                           'fontSize': '1.1em',
                           'marginBottom': '30px'
                       })
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),

            # Control panel with improved styling
            html.Div([
                html.Div([
                    html.Label("Data File Upload:",
                              style={'fontWeight': 'bold', 'color': '#34495e', 'marginBottom': '10px'}),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className='fas fa-upload', style={'marginRight': '8px'}),
                            'Drag and Drop or ',
                            html.A('Select CSV File', style={'color': '#3498db', 'textDecoration': 'underline'})
                        ]),
                        style={
                            'width': '100%', 'height': '70px', 'lineHeight': '70px',
                            'borderWidth': '2px', 'borderStyle': 'dashed',
                            'borderRadius': '8px', 'textAlign': 'center',
                            'margin': '10px 0', 'borderColor': '#bdc3c7',
                            'backgroundColor': '#ffffff',
                            'transition': 'all 0.3s ease'
                        },
                        multiple=False
                    )
                ], className='four columns', style={'padding': '0 15px'}),

                html.Div([
                    html.Label("Time Window (seconds):",
                              style={'fontWeight': 'bold', 'color': '#34495e', 'marginBottom': '10px'}),
                    dcc.Slider(
                        id='time-window',
                        min=1, max=30, step=1, value=self.config['window_size'],
                        marks={i: str(i) for i in range(1, 31, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className='four columns', style={'padding': '0 15px'}),

                html.Div([
                    html.Label("Start Time (seconds):",
                              style={'fontWeight': 'bold', 'color': '#34495e', 'marginBottom': '10px'}),
                    dcc.Slider(
                        id='start-time',
                        min=0, max=100, step=0.1, value=0,
                        marks={}, tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className='four columns', style={'padding': '0 15px'}),
            ], className='row', style={
                'margin': '20px 0',
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),

            # Channel selection with improved styling
            html.Div([
                html.Div([
                    html.Label("EEG Channels (µV):",
                              style={'fontWeight': 'bold', 'color': '#2980b9', 'marginBottom': '15px', 'fontSize': '1.1em'}),
                    dcc.Checklist(
                        id='eeg-channels',
                        options=[],
                        value=[],
                        style={'columnCount': 2, 'fontSize': '0.95em'},
                        inputStyle={'marginRight': '8px'},
                        labelStyle={'display': 'block', 'marginBottom': '8px', 'color': '#34495e'}
                    )
                ], className='four columns', style={'padding': '0 15px'}),

                html.Div([
                    html.Label("ECG Channels (mV):",
                              style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': '15px', 'fontSize': '1.1em'}),
                    dcc.Checklist(
                        id='ecg-channels',
                        options=[],
                        value=[],
                        style={'columnCount': 1, 'fontSize': '0.95em'},
                        inputStyle={'marginRight': '8px'},
                        labelStyle={'display': 'block', 'marginBottom': '8px', 'color': '#34495e'}
                    )
                ], className='four columns', style={'padding': '0 15px'}),

                html.Div([
                    html.Label("Reference (CM):",
                              style={'fontWeight': 'bold', 'color': '#95a5a6', 'marginBottom': '15px', 'fontSize': '1.1em'}),
                    dcc.Checklist(
                        id='cm-channels',
                        options=[],
                        value=[],
                        style={'columnCount': 1, 'fontSize': '0.95em'},
                        inputStyle={'marginRight': '8px'},
                        labelStyle={'display': 'block', 'marginBottom': '8px', 'color': '#34495e'}
                    )
                ], className='four columns', style={'padding': '0 15px'}),
            ], className='row', style={
                'margin': '20px 0',
                'backgroundColor': '#ffffff',
                'padding': '25px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),

            # Export controls with improved styling
            html.Div([
                html.Button('📊 Export Current View', id='export-btn',
                           style={
                               'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                               'padding': '12px 24px', 'margin': '10px', 'borderRadius': '6px',
                               'fontSize': '1em', 'fontWeight': 'bold', 'cursor': 'pointer',
                               'transition': 'all 0.3s ease'
                           }),
                html.Button('💾 Export Full Data', id='export-full-btn',
                           style={
                               'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none',
                               'padding': '12px 24px', 'margin': '10px', 'borderRadius': '6px',
                               'fontSize': '1em', 'fontWeight': 'bold', 'cursor': 'pointer',
                               'transition': 'all 0.3s ease'
                           }),
                html.Div(id='export-status', style={
                    'margin': '15px', 'fontSize': '1em', 'fontWeight': 'bold',
                    'color': '#27ae60', 'textAlign': 'center'
                })
            ], style={
                'textAlign': 'center', 'margin': '20px 0',
                'backgroundColor': '#ffffff', 'padding': '20px',
                'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),

            # Main plot with improved styling
            html.Div([
                dcc.Graph(id='signal-plot', style={'height': '650px'})
            ], style={
                'margin': '20px 0', 'backgroundColor': '#ffffff',
                'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'padding': '15px'
            }),

            # Statistics panel
            html.Div(id='statistics', style={
                'margin': '20px 0', 'backgroundColor': '#ffffff',
                'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'padding': '20px'
            }),

            # Hidden div to store data
            html.Div(id='data-store', style={'display': 'none'})
        ], style={
            'backgroundColor': '#ecf0f1', 'minHeight': '100vh',
            'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
        })

    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""

        @self.app.callback(
            [Output('data-store', 'children'),
             Output('eeg-channels', 'options'),
             Output('eeg-channels', 'value'),
             Output('ecg-channels', 'options'),
             Output('ecg-channels', 'value'),
             Output('cm-channels', 'options'),
             Output('cm-channels', 'value'),
             Output('start-time', 'max'),
             Output('start-time', 'marks')],
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename')]
        )
        def update_data(contents, filename):
            if contents is None:
                if self.data is not None:
                    # Use pre-loaded data
                    max_time = self.data['Time'].max()
                    marks = {i: f'{i}s' for i in range(0, int(max_time), max(1, int(max_time/10)))}

                    eeg_options = [{'label': ch, 'value': ch} for ch in self.eeg_channels]
                    ecg_options = [{'label': ch, 'value': ch} for ch in self.ecg_channels]
                    cm_options = [{'label': ch, 'value': ch} for ch in self.cm_channels]

                    return (self.data.to_json(date_format='iso', orient='split'),
                            eeg_options, self.eeg_channels[:4],  # Select first 4 EEG by default
                            ecg_options, self.ecg_channels,
                            cm_options, [],  # CM channels off by default
                            max_time, marks)
                return (None, [], [], [], [], [], [], 100, {})

            # Parse uploaded file
            import base64
            import io

            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            try:
                # Handle uploaded file with QUASAR format
                content_string = decoded.decode('utf-8')
                lines = content_string.split('\n')
                data_lines = [line for line in lines if not line.strip().startswith('#')]
                filtered_content = '\n'.join(data_lines)

                # Extract sampling rate from header if present
                temp_sampling_rate = self.sampling_rate  # Default
                for line in lines:
                    if line.startswith('#') and 'Sample_Frequency_(Hz)' in line:
                        try:
                            freq_value = line.split('=')[1].split(',')[1].strip()
                            temp_sampling_rate = float(freq_value)
                        except:
                            pass

                df = pd.read_csv(io.StringIO(filtered_content))

                # Process using QUASAR channel detection
                ignore_columns = ['Time', 'time', 'timestamp', 'Trigger', 'Time_Offset', 'ADC_Status',
                                'ADC_Sequence', 'Event', 'Comments', 'X3:']
                channels = [col for col in df.columns
                           if not any(ignore in col for ignore in ignore_columns)]

                # QUASAR channel classification
                eeg_names = ['Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4', 'Fp1', 'Fp2',
                            'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'A1', 'A2', 'Pz']
                ecg_names = ['X1:LEOG', 'X2:REOG']

                eeg_chs = [ch for ch in channels if ch in eeg_names]
                ecg_chs = [ch for ch in channels if ch in ecg_names]
                cm_chs = [ch for ch in channels if ch == 'CM']

                if 'Time' not in df.columns and 'time' not in df.columns:
                    df['Time'] = np.arange(len(df)) / temp_sampling_rate
                elif 'time' in df.columns:
                    df['Time'] = df['time']

                max_time = df['Time'].max()
                marks = {i: f'{i}s' for i in range(0, int(max_time), max(1, int(max_time/10)))}

                eeg_options = [{'label': ch, 'value': ch} for ch in eeg_chs]
                ecg_options = [{'label': ch, 'value': ch} for ch in ecg_chs]
                cm_options = [{'label': ch, 'value': ch} for ch in cm_chs]

                return (df.to_json(date_format='iso', orient='split'),
                        eeg_options, eeg_chs[:4],  # Select first 4 EEG by default
                        ecg_options, ecg_chs,
                        cm_options, [],  # CM channels off by default
                        max_time, marks)

            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                return (None, [], [], [], [], [], [], 100, {})

        @self.app.callback(
            [Output('signal-plot', 'figure'),
             Output('statistics', 'children')],
            [Input('data-store', 'children'),
             Input('eeg-channels', 'value'),
             Input('ecg-channels', 'value'),
             Input('cm-channels', 'value'),
             Input('time-window', 'value'),
             Input('start-time', 'value')]
        )
        def update_plot(data_json, selected_eeg, selected_ecg, selected_cm, time_window, start_time):
            try:
                if data_json is None:
                    return {}, "⚠️ No data loaded. Please upload a CSV file."

                # Validate inputs
                if not isinstance(time_window, (int, float)) or time_window <= 0:
                    return {}, "⚠️ Invalid time window value"

                if not isinstance(start_time, (int, float)) or start_time < 0:
                    return {}, "⚠️ Invalid start time value"

                # Load data from JSON
                try:
                    df = pd.read_json(data_json, orient='split')
                except ValueError as e:
                    return {}, f"⚠️ Error loading data: {str(e)}"

                if df.empty:
                    return {}, "⚠️ Dataset is empty"

                # Check if Time column exists
                if 'Time' not in df.columns:
                    return {}, "⚠️ No time column found in data"

                # Filter data by time window
                end_time = start_time + time_window

                # Validate time bounds
                max_time = df['Time'].max()
                if start_time >= max_time:
                    return {}, f"⚠️ Start time ({start_time:.1f}s) exceeds data duration ({max_time:.1f}s)"

                mask = (df['Time'] >= start_time) & (df['Time'] <= end_time)
                df_window = df[mask]

                if df_window.empty:
                    return {}, f"⚠️ No data in time window {start_time:.1f}s - {end_time:.1f}s"

                # Check if any channels are selected
                all_selected = (selected_eeg or []) + (selected_ecg or []) + (selected_cm or [])
                if not all_selected:
                    return {}, "ℹ️ Please select at least one channel to display"

            # Create subplots with multiple y-axes for different scales
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )

            # QUASAR-specific scaling:
            # EEG channels: typically tens to hundreds of µV
            # ECG channels (X1:LEOG, X2:REOG): thousands of µV (~mV range)
            # CM: large amplitude reference signal

            # Add EEG traces (µV scale) - QUASAR data is already in µV
            for i, channel in enumerate(selected_eeg or []):
                if channel in df_window.columns:
                    # QUASAR data is already in µV, no conversion needed
                    y_data = df_window[channel]
                    fig.add_trace(
                        go.Scatter(
                            x=df_window['Time'],
                            y=y_data,
                            mode='lines',
                            name=f'{channel} (µV)',
                            line=dict(color=px.colors.qualitative.Set1[i % 10]),
                            yaxis='y'
                        )
                    )

            # Add ECG traces (mV scale) - QUASAR data is in µV, convert to mV for display
            for i, channel in enumerate(selected_ecg or []):
                if channel in df_window.columns:
                    # Convert µV to mV for ECG display (QUASAR ECG channels are high amplitude)
                    y_data = df_window[channel] / 1000  # Convert µV to mV
                    fig.add_trace(
                        go.Scatter(
                            x=df_window['Time'],
                            y=y_data,
                            mode='lines',
                            name=f'{channel} (mV)',
                            line=dict(color=px.colors.qualitative.Set2[i % 8], dash='dash'),
                            yaxis='y2'
                        ),
                        secondary_y=True
                    )

            # Add CM traces (reference signal) - plot on secondary axis due to large amplitude
            for i, channel in enumerate(selected_cm or []):
                if channel in df_window.columns:
                    # CM is reference signal with large amplitude, convert µV to mV for display
                    y_data = df_window[channel] / 1000  # Convert µV to mV for consistency
                    fig.add_trace(
                        go.Scatter(
                            x=df_window['Time'],
                            y=y_data,
                            mode='lines',
                            name=f'{channel} (Reference)',
                            line=dict(color='gray', width=1, dash='dot'),
                            yaxis='y2',
                            opacity=0.7
                        ),
                        secondary_y=True
                    )

            # Update layout
            signal_types = []
            if selected_eeg: signal_types.append("EEG")
            if selected_ecg: signal_types.append("ECG")
            if selected_cm: signal_types.append("Reference")

            title = f'QUASAR {"/".join(signal_types)} Signals ({start_time:.1f}s - {end_time:.1f}s)'

            fig.update_layout(
                title=title,
                xaxis_title='Time (s)',
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                height=650,
                margin=dict(l=60, r=60, t=80, b=60)
            )

            # Set y-axes titles and ranges
            fig.update_yaxes(title_text="EEG Amplitude (µV)", secondary_y=False)
            fig.update_yaxes(title_text="ECG/Reference Amplitude (mV)", secondary_y=True)

            # Calculate statistics
            stats_data = []
            all_selected = (selected_eeg or []) + (selected_ecg or []) + (selected_cm or [])

            for channel in all_selected:
                if channel in df_window.columns:
                    data_vals = df_window[channel].dropna()
                    if not data_vals.empty:
                        # Determine units for display - QUASAR data is already in µV
                        if channel in (selected_eeg or []):
                            units = "µV"
                            display_vals = data_vals  # Already in µV
                        else:
                            units = "mV"
                            display_vals = data_vals / 1000  # Convert µV to mV

                        stats_data.append({
                            'Channel': channel,
                            'Units': units,
                            'Mean': f"{display_vals.mean():.2f}",
                            'Std': f"{display_vals.std():.2f}",
                            'Min': f"{display_vals.min():.2f}",
                            'Max': f"{display_vals.max():.2f}",
                            'Samples': len(data_vals)
                        })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_table = html.Table([
                    html.Thead([
                        html.Tr([html.Th(col) for col in stats_df.columns])
                    ]),
                    html.Tbody([
                        html.Tr([html.Td(stats_df.iloc[i][col]) for col in stats_df.columns])
                        for i in range(len(stats_df))
                    ])
                ], style={'margin': 'auto', 'width': '90%'})

                stats_component = html.Div([
                    html.H3("Signal Statistics"),
                    stats_table
                ])
            else:
                stats_component = "No statistics available"

                return fig, stats_component

            except Exception as e:
                logger.error(f"Error in update_plot: {str(e)}")
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Error in Plot Generation",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    annotations=[{
                        'text': f"⚠️ Plot generation failed: {str(e)}",
                        'x': 0.5, 'y': 0.5,
                        'xref': 'paper', 'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': 'red'}
                    }]
                )
                return empty_fig, f"⚠️ Error: {str(e)}"

        @self.app.callback(
            Output('export-status', 'children'),
            [Input('export-btn', 'n_clicks'),
             Input('export-full-btn', 'n_clicks')],
            [State('data-store', 'children'),
             State('eeg-channels', 'value'),
             State('ecg-channels', 'value'),
             State('cm-channels', 'value'),
             State('time-window', 'value'),
             State('start-time', 'value')]
        )
        def export_data(export_clicks, export_full_clicks, data_json, selected_eeg, selected_ecg, selected_cm, time_window, start_time):
            if not callback_context.triggered:
                return ""

            if data_json is None:
                return "No data to export"

            try:
                df = pd.read_json(data_json, orient='split')
                button_clicked = callback_context.triggered[0]['prop_id'].split('.')[0]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if button_clicked == 'export-btn':
                    # Export current view
                    end_time = start_time + time_window
                    mask = (df['Time'] >= start_time) & (df['Time'] <= end_time)
                    df_export = df[mask]
                    filename = f"eeg_export_view_{timestamp}.csv"
                elif button_clicked == 'export-full-btn':
                    # Export full data
                    df_export = df
                    filename = f"eeg_export_full_{timestamp}.csv"

                # Filter selected channels
                channels_to_export = ['Time'] + (selected_eeg or []) + (selected_ecg or []) + (selected_cm or [])
                df_export = df_export[channels_to_export]

                # Save to file
                output_path = os.path.join(os.getcwd(), filename)
                df_export.to_csv(output_path, index=False)

                return f"Data exported to {filename} ({len(df_export)} rows, {len(df_export.columns)} columns)"

            except Exception as e:
                return f"Export failed: {str(e)}"

    def run(self, debug=False, host='127.0.0.1', port=8050):
        """Run the Dash application."""
        logger.info(f"Starting EEG Viewer on http://{host}:{port}")
        self.app.run(debug=debug, host=host, port=port)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='QUASAR EEG/ECG Data Visualization Tool')
    parser.add_argument('--data', '-d', type=str, help='CSV data file path')
    parser.add_argument('--config', '-c', type=str, help='Configuration JSON file path')
    parser.add_argument('--host', default='127.0.0.1', help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8050, help='Port number (default: 8050)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    # Create viewer instance
    viewer = EEGViewer(data_file=args.data, config_file=args.config)

    # Run the application
    viewer.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()