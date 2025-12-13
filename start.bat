@echo off
REM Network Traffic Classification - Main Launcher

:menu
cls
echo ======================================================================
echo Network Traffic Classification System
echo ======================================================================
echo.
echo Select an option:
echo.
echo   [1] Setup (First time installation)
echo   [2] Run Web Dashboard
echo   [3] Retrain Models
echo   [4] Run Tests
echo   [5] Exit
echo.
echo ======================================================================
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto dashboard
if "%choice%"=="3" goto retrain
if "%choice%"=="4" goto tests
if "%choice%"=="5" goto end

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:setup
cls
echo Starting setup...
call setup.bat
goto menu

:dashboard
cls
echo Starting Dashboard...
call run_dashboard.bat
goto menu

:retrain
cls
echo Retraining models...
call retrain.bat
goto menu

:tests
cls
echo Running tests...
call run_tests.bat
goto menu

:end
cls
echo.
echo Thank you for using Network Traffic Classification System!
echo.
timeout /t 2 >nul
exit
