@echo off
REM Network Traffic Classification - Retrain Models

echo ======================================================================
echo Retraining Network Traffic Classification Models
echo ======================================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

REM Backup existing models
if exist "models\rf_model.pkl" (
    echo Backing up existing models...
    if not exist "models\backup" mkdir models\backup
    xcopy /Y models\*.pkl models\backup\ >nul 2>&1
    echo Backup created in models\backup\
    echo.
)

REM Train models
echo Training models...
echo This will take 1-2 minutes...
echo.
python train.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Retraining Complete!
echo ======================================================================
echo.
echo New models have been saved to models/
echo Previous models backed up to models/backup/
echo.
pause
