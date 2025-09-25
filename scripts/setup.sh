#!/bin/bash

# QUASAR EEG Viewer Setup Script
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on any error

echo "🧠 QUASAR EEG Viewer Setup"
echo "=========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📁 Project directory: $PROJECT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old environment..."
    rm -rf venv
fi

python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data config logs exports

# Generate sample data
echo "🧪 Generating sample data..."
python scripts/generate_sample_data.py

# Create .gitkeep files for empty directories
touch data/.gitkeep
touch logs/.gitkeep
touch exports/.gitkeep

# Set executable permissions
chmod +x src/eeg_viewer.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the EEG Viewer:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the application: python src/eeg_viewer.py"
echo "  3. Open your browser to: http://127.0.0.1:8050"
echo ""
echo "For more information, see README.md"

# Optional: Test the installation
read -p "Would you like to run a quick test? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running quick test..."
    python -c "
import pandas as pd
import plotly
import dash
print('✅ All core dependencies imported successfully')
print(f'   - pandas: {pd.__version__}')
print(f'   - plotly: {plotly.__version__}')
print(f'   - dash: {dash.__version__}')
"
    echo "✅ Test completed!"
fi

echo ""
echo "Setup script completed. Happy analyzing! 🧠📊"