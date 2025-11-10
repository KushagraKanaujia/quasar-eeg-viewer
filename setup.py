"""
Setup configuration for EEG/ECG Multichannel Viewer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eeg-ecg-viewer",
    version="1.0.0",
    author="Kushagra Kanaujia",
    author_email="KushagraKanaujia@users.noreply.github.com",
    description="An interactive visualization tool for multi-channel EEG and ECG signal analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KushagraKanaujia/quasar-eeg-viewer",
    project_urls={
        "Bug Tracker": "https://github.com/KushagraKanaujia/quasar-eeg-viewer/issues",
        "Documentation": "https://github.com/KushagraKanaujia/quasar-eeg-viewer#readme",
        "Source Code": "https://github.com/KushagraKanaujia/quasar-eeg-viewer",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eeg-viewer=eeg_ecg_plotter:main",
        ],
    },
    keywords=[
        "eeg",
        "ecg",
        "signal-processing",
        "visualization",
        "neuroscience",
        "biomedical",
        "plotly",
        "interactive",
    ],
)
