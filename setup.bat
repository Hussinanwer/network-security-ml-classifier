@echo off
REM Network Traffic Classification - Setup Script
REM This script sets up the environment and trains the models

echo ======================================================================
echo Network Traffic Classification - Setup
echo ======================================================================
echo.

REM Check Python installation
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

python --version
echo Python found!
echo.

REM Create virtual environment (optional but recommended)
echo [2/4] Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists!
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo [3/4] Installing dependencies...
echo This may take a few minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env >nul
    echo .env file created!
) else (
    echo .env file already exists!
)
echo.

REM Train models
echo [4/4] Training models...
echo This will take 1-2 minutes...
python train.py
if %errorlevel% neq 0 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)
echo.

REM Verify models were created
if not exist "models\rf_model.pkl" (
    echo ERROR: Model files were not created
    pause
    exit /b 1
)

echo ======================================================================
echo Setup Complete!
echo ======================================================================
echo.
echo Models have been trained and saved to the models/ directory.
echo.
echo Next steps:
echo   1. Run the API:       run_api.bat
echo   2. Run the Dashboard: run_dashboard.bat
echo   3. Run tests:         run_tests.bat
echo.
echo To use the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
pause
