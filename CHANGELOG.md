# Changelog

All notable changes to the EEG/ECG Multichannel Viewer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-09-26

### Added
- Initial release of EEG/ECG Multichannel Viewer
- Interactive visualization with Plotly
- Dual y-axis scaling for EEG (µV) and ECG (mV) signals
- Automatic channel classification based on standard 10-20 EEG nomenclature
- Pan, zoom, and scroll capabilities
- Built-in range slider for temporal navigation
- Channel toggling via legend
- High-resolution PNG export functionality
- Command-line interface with `--data` and `--output` options
- Python API for programmatic use
- Support for CSV files with comment line filtering
- Comprehensive documentation:
  - README with features and installation instructions
  - USAGE guide with examples
  - API reference documentation
  - Contributing guidelines
  - Examples directory with sample use cases
- GitHub Actions CI/CD workflow
- MIT License
- Professional project structure with setup.py

### Features
- **Data Loading**: Robust CSV parsing with automatic comment filtering
- **Channel Classification**: Supports 21 standard EEG positions, ECG/EOG channels, and reference channels
- **Interactive Controls**:
  - Unified hover showing synchronized values
  - Box select and lasso tools
  - Drawing tools for annotations
  - Zoom reset and autoscale
- **Visualization**:
  - Color-coded channels by type
  - Grouped legend organization
  - Distinct line styles (solid/dashed/dotted)
  - Responsive layout
- **Export**: Configurable PNG export (1200×700px, 2× scale)

### Technical
- Python 3.8+ support
- Compatible with pandas 1.5.0+, plotly 5.15.0+, numpy 1.21.0+
- Cross-platform (Windows, macOS, Linux)
- Standalone HTML output (no server required)
- Memory-efficient data processing

## [Unreleased]

### Planned
- Digital filtering capabilities (bandpass, notch filters)
- Spectral analysis and power spectral density plots
- Time-frequency analysis (spectrograms)
- Real-time streaming data support
- EDF file format support
- Annotation and event marking tools
- Multi-file comparison mode
- Configurable color schemes
- Automated artifact detection
- Statistical analysis features
- Unit test suite
- Performance benchmarks
- Docker containerization
- Web application version

---

## Version Numbering

This project follows Semantic Versioning:
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md with release date and changes
3. Create git tag: `git tag -a v1.0.0 -m "Release version 1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release with changelog excerpt
