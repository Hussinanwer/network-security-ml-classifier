# Windows Setup Guide

Complete setup guide for Windows users.

## Prerequisites

### 1. Install Python

1. Download Python 3.10+ from https://www.python.org/downloads/
2. **IMPORTANT**: During installation, check "Add Python to PATH"
3. Verify installation:
   ```cmd
   python --version
   ```
   Should show: `Python 3.10.x` or higher

### 2. Install Git (Optional)

Only needed if cloning from a repository:
- Download from https://git-scm.com/download/win

## Installation Methods

### Method 1: Automatic Installation (Recommended)

**Easiest way to get started:**

1. Navigate to the project folder
2. Double-click `INSTALL.bat`
3. Wait 5-10 minutes for installation and training
4. Done! The system is ready to use

### Method 2: Using start.bat Menu

1. Double-click `start.bat`
2. Select option `[1] Setup`
3. Wait for completion
4. Use the menu to launch services

### Method 3: Manual Installation

Open Command Prompt in the project folder and run:

```cmd
REM Install dependencies
pip install -r requirements.txt

REM Create environment file
copy .env.example .env

REM Train models
python train.py
```

## Running the Application

### Option 1: Interactive Menu (Easiest)

Double-click `start.bat` and select:

```
======================================================================
Network Traffic Classification System
======================================================================

Select an option:

  [1] Setup (First time installation)
  [2] Run REST API
  [3] Run Web Dashboard
  [4] Run Both (API + Dashboard)
  [5] Retrain Models
  [6] Run Tests
  [7] Open API Documentation
  [8] Exit

======================================================================
```

### Option 2: Individual Batch Files

**Run REST API:**
- Double-click `run_api.bat`
- Access: http://localhost:8000/docs

**Run Web Dashboard:**
- Double-click `run_dashboard.bat`
- Opens automatically in browser: http://localhost:8501

**Run Both Services:**
- From `start.bat`, select option `[4] Run Both`
- Both services open in separate windows

## Batch Files Reference

| File | Purpose |
|------|---------|
| `INSTALL.bat` | Complete first-time installation |
| `start.bat` | Interactive menu for all operations |
| `setup.bat` | Setup environment and train models |
| `run_api.bat` | Start REST API server |
| `run_dashboard.bat` | Start Web Dashboard |
| `retrain.bat` | Retrain models with backup |
| `run_tests.bat` | Run test suite |

## Common Tasks

### Making Predictions via API

1. Start API: `run_api.bat`
2. Open browser: http://localhost:8000/docs
3. Try the `/predict` endpoint with sample data

### Using the Dashboard

1. Start Dashboard: `run_dashboard.bat`
2. Browser opens automatically
3. Choose:
   - **Single Prediction**: Enter features manually
   - **Batch Prediction**: Upload CSV file

### Retraining Models

1. Double-click `retrain.bat`
2. Old models are automatically backed up
3. New models are trained and saved

### Running Tests

1. Double-click `run_tests.bat`
2. View test results in console

## Virtual Environment

The setup creates a virtual environment in the `venv` folder.

**To activate manually:**
```cmd
venv\Scripts\activate.bat
```

**To deactivate:**
```cmd
deactivate
```

**Benefits:**
- Isolated dependencies
- No conflicts with other Python projects
- Easy to delete and recreate

## Troubleshooting

### "Python is not recognized"

**Problem:** Python not in PATH

**Solution:**
1. Reinstall Python
2. Check "Add Python to PATH" during installation
3. Or manually add Python to PATH:
   - Search "Environment Variables" in Windows
   - Edit PATH variable
   - Add Python installation folder

### "Module not found" errors

**Problem:** Dependencies not installed

**Solution:**
```cmd
pip install -r requirements.txt
```

### Port already in use

**Problem:** Port 8000 or 8501 is occupied

**Solution:**
1. Edit `.env` file
2. Change `API_PORT` or `DASHBOARD_PORT`
3. Restart the service

### Models not found

**Problem:** Models not trained

**Solution:**
```cmd
python train.py
```
Or run `setup.bat`

### Permission errors

**Problem:** Antivirus blocking

**Solution:**
1. Add project folder to antivirus exceptions
2. Run Command Prompt as Administrator

### Virtual environment issues

**Problem:** venv corrupted

**Solution:**
```cmd
REM Delete old venv
rmdir /s /q venv

REM Create new venv
python -m venv venv

REM Activate and reinstall
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Firewall Configuration

When running for the first time, Windows Firewall may prompt:

1. **Allow access** when prompted
2. Both API and Dashboard need network access
3. Or manually add exception in Windows Firewall settings

## Performance Tips

### Speed up training:
- Use SSD instead of HDD
- Close other applications
- Increase `RF_N_ESTIMATORS` in `.env` for better accuracy (slower training)

### Speed up API:
- Run on SSD
- Increase RAM allocation
- Use production ASGI server (already configured)

## Uninstalling

To completely remove:

1. Delete the project folder
2. (Optional) Remove Python if not needed for other projects

To reset and start fresh:

1. Delete `venv` folder
2. Delete `models` folder
3. Delete `.env` file
4. Run `INSTALL.bat` again

## Next Steps

After successful installation:

1. **Learn the API**: Read `API.md`
2. **Try the Dashboard**: Explore single and batch predictions
3. **Read Documentation**: Check `README.md` and `QUICKSTART.md`
4. **Experiment**: Try different models, retrain with different parameters

## Getting Help

If you encounter issues:

1. Check error messages in console
2. Review this guide
3. Check `README.md` for detailed information
4. Ensure Python 3.10+ is installed
5. Verify all dependencies are installed

## Directory Structure After Installation

```
project/
├── venv/                    # Virtual environment (created)
├── models/                  # Trained models (created)
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   ├── svm_model.pkl
│   └── ...
├── api/
├── tests/
├── .env                     # Config file (created)
├── INSTALL.bat             # Run this first
├── start.bat               # Interactive menu
├── run_api.bat
├── run_dashboard.bat
└── ...
```

## Support

For questions or issues:
- Review error messages
- Check troubleshooting section above
- Ensure prerequisites are met
- Verify files are not corrupted
