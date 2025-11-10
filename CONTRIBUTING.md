# Contributing to EEG/ECG Multichannel Viewer

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, library versions)
- Sample data or code snippet if applicable

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach (optional)

### Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

3. **Test your changes**:
   - Ensure the code runs without errors
   - Test with sample EEG/ECG data
   - Verify HTML output displays correctly

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

## Code Style

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions using Google style format:
  ```python
  def function_name(arg1, arg2):
      """
      Brief description of function.

      Args:
          arg1 (type): Description of arg1
          arg2 (type): Description of arg2

      Returns:
          type: Description of return value
      """
  ```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/KushagraKanaujia/quasar-eeg-viewer.git
cd quasar-eeg-viewer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install black flake8 pytest
```

## Testing

Before submitting a pull request:
1. Test with various CSV data formats
2. Verify output HTML renders correctly in multiple browsers
3. Check for Python errors and warnings
4. Ensure backward compatibility

## Areas for Contribution

We welcome contributions in these areas:

### Signal Processing
- Digital filtering implementations (bandpass, notch filters)
- Artifact detection algorithms
- Spectral analysis features
- Time-frequency analysis

### Visualization
- Additional plot types (spectrograms, topographic maps)
- Customizable color schemes
- Annotation tools for marking events
- Multi-file comparison views

### Data Format Support
- EDF (European Data Format) file support
- BIDS (Brain Imaging Data Structure) compliance
- Additional CSV format handling
- Real-time streaming data support

### User Interface
- Command-line interface improvements
- Configuration file support
- Interactive parameter adjustment
- Export format options

### Documentation
- Tutorial notebooks
- Example datasets
- API documentation
- Video demonstrations

### Testing
- Unit tests for data processing functions
- Integration tests for end-to-end workflows
- Performance benchmarking
- Cross-platform testing

## Questions?

Feel free to open an issue for:
- Clarification on contribution guidelines
- Discussion of proposed features
- Help with development setup
- General questions about the project

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other contributors

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
