@echo off
REM QUASAR EEG Viewer Setup Script for Windows
REM This script sets up the Python environment and installs all dependencies

echo 🧠 QUASAR EEG Viewer Setup
echo ==========================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check Python version (simplified check)
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% found

REM Get the directory where the script is located
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

echo 📁 Project directory: %PROJECT_DIR%

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Create virtual environment
echo 🔧 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing old environment...
    rmdir /s /q venv
)

python -m venv venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directory structure...
if not exist data mkdir data
if not exist config mkdir config
if not exist logs mkdir logs
if not exist exports mkdir exports

REM Generate sample data
echo 🧪 Generating sample data...
python scripts\generate_sample_data.py

REM Create .gitkeep files for empty directories
type nul > data\.gitkeep
type nul > logs\.gitkeep
type nul > exports\.gitkeep

echo.
echo 🎉 Setup completed successfully!
echo.
echo To start the EEG Viewer:
echo   1. Activate the virtual environment: venv\Scripts\activate.bat
echo   2. Run the application: python src\eeg_viewer.py
echo   3. Open your browser to: http://127.0.0.1:8050
echo.
echo For more information, see README.md

REM Optional: Test the installation
set /p "test_choice=Would you like to run a quick test? (y/n): "
if /i "%test_choice%"=="y" (
    echo 🧪 Running quick test...
    python -c "import pandas as pd; import plotly; import dash; print('✅ All core dependencies imported successfully'); print(f'   - pandas: {pd.__version__}'); print(f'   - plotly: {plotly.__version__}'); print(f'   - dash: {dash.__version__}')"
    echo ✅ Test completed!
)

echo.
echo Setup script completed. Happy analyzing! 🧠📊
pause