#!/usr/bin/env python3
"""
Sample EEG/ECG Data Generator

Generates realistic synthetic EEG and ECG data for testing the QUASAR EEG Viewer.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


def generate_eeg_signal(duration_s, sampling_rate, noise_level=0.1):
    """Generate realistic EEG signal with multiple frequency components."""
    t = np.linspace(0, duration_s, int(duration_s * sampling_rate), False)

    # EEG frequency bands (Hz)
    # Delta: 0.5-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz, Gamma: 30-100 Hz

    signal = (
        # Alpha waves (8-13 Hz) - most prominent in occipital regions
        20e-6 * np.sin(2 * np.pi * 10 * t) +
        15e-6 * np.sin(2 * np.pi * 11 * t + np.pi/4) +

        # Beta waves (13-30 Hz) - frontal and central regions
        8e-6 * np.sin(2 * np.pi * 20 * t + np.pi/3) +
        6e-6 * np.sin(2 * np.pi * 25 * t) +

        # Theta waves (4-8 Hz) - temporal regions
        12e-6 * np.sin(2 * np.pi * 6 * t + np.pi/6) +

        # Delta waves (0.5-4 Hz) - deep sleep patterns
        25e-6 * np.sin(2 * np.pi * 2 * t) +

        # High-frequency noise and artifacts
        noise_level * 5e-6 * np.random.normal(0, 1, len(t))
    )

    return t, signal


def generate_ecg_signal(duration_s, sampling_rate, heart_rate=75):
    """Generate realistic ECG signal with P-QRS-T complexes."""
    t = np.linspace(0, duration_s, int(duration_s * sampling_rate), False)

    # Heart rate in beats per second
    beat_frequency = heart_rate / 60.0

    signal = np.zeros_like(t)

    # Generate individual heartbeats
    beat_period = 1.0 / beat_frequency

    for beat_start in np.arange(0, duration_s, beat_period):
        beat_indices = (t >= beat_start) & (t < beat_start + beat_period)
        beat_time = t[beat_indices] - beat_start

        if len(beat_time) > 0:
            # P wave (atrial depolarization)
            p_wave = 0.2e-3 * np.exp(-((beat_time - 0.1)**2) / (2 * 0.02**2))

            # QRS complex (ventricular depolarization)
            qrs_time = beat_time - 0.2
            qrs_wave = np.where(
                (qrs_time >= -0.02) & (qrs_time <= 0.02),
                1.5e-3 * np.sin(np.pi * qrs_time / 0.04),
                0
            )

            # T wave (ventricular repolarization)
            t_wave = 0.4e-3 * np.exp(-((beat_time - 0.35)**2) / (2 * 0.08**2))

            signal[beat_indices] += p_wave + qrs_wave + t_wave

    # Add noise
    signal += 0.05e-3 * np.random.normal(0, 1, len(t))

    return t, signal


def generate_sample_dataset(duration_s=60, sampling_rate=1000):
    """Generate a complete sample dataset with multiple EEG and ECG channels."""

    print(f"Generating {duration_s}s of sample data at {sampling_rate}Hz...")

    # Time vector
    t = np.linspace(0, duration_s, int(duration_s * sampling_rate), False)

    # Initialize data dictionary
    data = {'Time': t}

    # EEG channels (common 10-20 system locations)
    eeg_channels = ['EEG_C3', 'EEG_C4', 'EEG_O1', 'EEG_O2', 'EEG_F3', 'EEG_F4', 'EEG_P3', 'EEG_P4']

    for i, channel in enumerate(eeg_channels):
        # Each channel has slightly different characteristics
        phase_shift = i * np.pi / 4
        amplitude_factor = 0.8 + 0.4 * np.random.random()
        noise_level = 0.1 + 0.05 * np.random.random()

        _, eeg_signal = generate_eeg_signal(duration_s, sampling_rate, noise_level)

        # Apply channel-specific modifications
        if 'O' in channel:  # Occipital - stronger alpha
            eeg_signal *= 1.5 * amplitude_factor
        elif 'F' in channel:  # Frontal - more beta activity
            _, beta_component = generate_eeg_signal(duration_s, sampling_rate, 0.05)
            eeg_signal = 0.7 * eeg_signal + 0.3 * beta_component
            eeg_signal *= amplitude_factor
        else:  # Central and parietal
            eeg_signal *= amplitude_factor

        # Add phase shift for spatial correlation
        eeg_signal = np.roll(eeg_signal, int(phase_shift * sampling_rate / (2 * np.pi)))

        data[channel] = eeg_signal

    # ECG channels (standard 12-lead ECG leads)
    ecg_channels = ['ECG_I', 'ECG_II', 'ECG_III']
    heart_rates = [72, 75, 73]  # Slightly different heart rates for each lead

    for i, (channel, hr) in enumerate(zip(ecg_channels, heart_rates)):
        _, ecg_signal = generate_ecg_signal(duration_s, sampling_rate, hr)

        # Each lead has different amplitude and polarity
        if channel == 'ECG_I':
            ecg_signal *= 1.0
        elif channel == 'ECG_II':
            ecg_signal *= 1.2  # Lead II typically has larger amplitude
        elif channel == 'ECG_III':
            ecg_signal *= 0.8

        data[channel] = ecg_signal

    return pd.DataFrame(data)


def main():
    """Main function to generate and save sample data."""

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    config_dir = Path(__file__).parent.parent / 'config'

    data_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    # Generate different datasets
    datasets = [
        {'name': 'sample_eeg_data.csv', 'duration': 120, 'sampling_rate': 1000},
        {'name': 'short_sample.csv', 'duration': 30, 'sampling_rate': 500},
        {'name': 'long_sample.csv', 'duration': 300, 'sampling_rate': 1000},
    ]

    for dataset in datasets:
        print(f"\nGenerating {dataset['name']}...")

        df = generate_sample_dataset(
            duration_s=dataset['duration'],
            sampling_rate=dataset['sampling_rate']
        )

        output_path = data_dir / dataset['name']
        df.to_csv(output_path, index=False)

        print(f"✅ Saved {len(df)} samples to {output_path}")
        print(f"   Channels: {', '.join(df.columns[1:])}")
        print(f"   Duration: {dataset['duration']}s")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Generate configuration file
    config = {
        "sampling_rate": 1000,
        "eeg_scale_factor": 1e-6,
        "ecg_scale_factor": 1e-3,
        "window_size": 10.0,
        "eeg_channels": ["EEG_C3", "EEG_C4", "EEG_O1", "EEG_O2", "EEG_F3", "EEG_F4", "EEG_P3", "EEG_P4"],
        "ecg_channels": ["ECG_I", "ECG_II", "ECG_III"],
        "colors": {
            "eeg": "#1f77b4",
            "ecg": "#d62728",
            "background": "#ffffff",
            "grid": "#f0f0f0"
        }
    }

    config_path = config_dir / 'viewer_config.json'
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ Configuration saved to {config_path}")
    print("\nSample data generation completed!")
    print("\nTo view the data:")
    print("  python src/eeg_viewer.py --data data/sample_eeg_data.csv")


if __name__ == '__main__':
    main()