@echo off
REM Network Traffic Classification - Run Dashboard

echo ======================================================================
echo Starting Network Traffic Classification Dashboard
echo ======================================================================
echo.

REM Check if models exist
if not exist "models\rf_model.pkl" (
    echo ERROR: Models not found!
    echo Please run setup.bat first to train the models.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

REM Start Dashboard
echo Starting Streamlit dashboard...
echo.
echo Dashboard will open automatically in your browser
echo.
echo If it doesn't open automatically, visit:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo ======================================================================
echo.

python run_dashboard.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Dashboard failed to start
    echo Check the error messages above
    echo.
)

pause
