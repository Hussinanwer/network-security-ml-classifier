@echo off
REM Network Traffic Classification - Run Tests

echo ======================================================================
echo Running Network Traffic Classification Tests
echo ======================================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

REM Run tests
echo Running test suite...
echo.
pytest tests/ -v

echo.
echo ======================================================================
echo Tests Complete!
echo ======================================================================
echo.
pause
