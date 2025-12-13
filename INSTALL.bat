@echo off
REM Network Traffic Classification - Quick Installer
REM Run this file for first-time setup

cls
echo ======================================================================
echo Network Traffic Classification - Quick Installer
echo ======================================================================
echo.
echo This will:
echo   1. Check Python installation
echo   2. Create virtual environment
echo   3. Install dependencies
echo   4. Train models
echo.
echo This may take 5-10 minutes depending on your system.
echo.
set /p confirm="Continue with installation? (Y/N): "

if /i not "%confirm%"=="Y" (
    echo Installation cancelled.
    pause
    exit /b 0
)

echo.
echo Starting installation...
echo.

REM Run setup
call setup.bat

if %errorlevel% neq 0 (
    echo.
    echo ======================================================================
    echo Installation Failed!
    echo ======================================================================
    echo.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

REM Success message
cls
echo ======================================================================
echo Installation Successful!
echo ======================================================================
echo.
echo Your Network Traffic Classification system is ready to use!
echo.
echo Quick Start:
echo.
echo   Double-click 'start.bat' to launch the menu, or use:
echo.
echo   - run_api.bat       : Start the REST API
echo   - run_dashboard.bat : Start the Web Dashboard
echo.
echo Documentation:
echo   - README.md         : Complete user guide
echo   - API.md            : API reference
echo   - QUICKSTART.md     : Quick start guide
echo.
echo ======================================================================
echo.

set /p launch="Would you like to launch the menu now? (Y/N): "

if /i "%launch%"=="Y" (
    start.bat
) else (
    echo.
    echo You can launch the menu anytime by running start.bat
    echo.
    pause
)

exit /b 0
