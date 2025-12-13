# Troubleshooting Guide

Common issues and their solutions.

## Dashboard Issues

### Error: "This site can't be reached" or "ERR_ADDRESS_INVALID"

**Problem:** Trying to access `http://0.0.0.0:8501`

**Solution:** Use `http://localhost:8501` instead

- `0.0.0.0` is a server bind address, not a browser URL
- Always use `localhost` or `127.0.0.1` in your browser
- The dashboard should open automatically with the correct URL

**Quick fix:**
1. Stop the dashboard (Ctrl+C)
2. Run `run_dashboard.bat` again
3. The browser should open automatically to the correct URL
4. Or manually visit: http://localhost:8501

---

### Dashboard doesn't open automatically

**Solution:**
1. Check the console for the URL
2. Manually open your browser
3. Navigate to: http://localhost:8501

---

### "Connection refused" error

**Possible causes:**

1. **Dashboard not running**
   - Run `run_dashboard.bat`
   - Wait for "Network URLs" message

2. **Port already in use**
   - Another program is using port 8501
   - Solution: Change port in `.env` file:
     ```
     DASHBOARD_PORT=8502
     ```
   - Restart dashboard

3. **Firewall blocking**
   - Allow Python through Windows Firewall
   - Or temporarily disable firewall to test

---

## Installation Issues

### "Python is not recognized"

**Problem:** Python not in system PATH

**Solution:**
1. Reinstall Python from https://www.python.org/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Restart Command Prompt
4. Verify: `python --version`

**Alternative:** Add Python manually to PATH
1. Find Python installation folder (e.g., `C:\Users\YourName\AppData\Local\Programs\Python\Python310`)
2. Add to System PATH:
   - Windows Key + Search "Environment Variables"
   - Edit "Path" variable
   - Add Python folder and Scripts folder
   - Click OK
3. Restart Command Prompt

---

### "No module named 'xyz'" error

**Problem:** Dependencies not installed

**Solution:**
```cmd
pip install -r requirements.txt
```

If using virtual environment:
```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

### Virtual environment issues

**Problem:** venv corrupted or not working

**Solution:** Recreate virtual environment
```cmd
REM Delete old venv
rmdir /s /q venv

REM Create new venv
python -m venv venv

REM Activate
venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt
```

---

## Model Training Issues

### Training takes too long

**Normal:** Training should take 1-2 minutes

**If taking longer:**
- Close other programs
- Check CPU usage (should be high during training)
- Reduce `RF_N_ESTIMATORS` in `.env` (default: 100)

---

### Training fails with memory error

**Solution:**
1. Close other programs
2. Reduce model complexity in `.env`:
   ```
   RF_N_ESTIMATORS=50
   ```
3. Restart computer if needed

---

### "Dataset not found" error

**Problem:** CSV file missing or in wrong location

**Solution:**
1. Verify `network_traffic_multiclass_dataset.csv` exists in project root
2. Check file name spelling (exact match required)
3. File should be in same folder as `train.py`

---

## Runtime Issues

### Predictions are slow

**Possible causes:**

1. **Model not loaded** (loading on each request)
   - Check API logs
   - Restart API

2. **CPU overloaded**
   - Close other programs
   - Check Task Manager

3. **Using wrong model**
   - Random Forest (100 trees) is slower but more accurate
   - For speed: Use Logistic Regression model

---

### Getting wrong predictions

**Checklist:**
1. Verify you're using the latest trained model
2. Check input features are in correct format
3. Ensure all 35 features are provided
4. IP addresses should be strings (e.g., "192.168.1.1")
5. Numeric values should be numbers, not strings

---

## Windows-Specific Issues

### Batch files won't run

**Solution 1:** Run from Command Prompt
```cmd
cd path\to\project
setup.bat
```

**Solution 2:** Unblock files
1. Right-click batch file
2. Properties
3. Check "Unblock" at bottom
4. Click OK

---

### "Access Denied" errors

**Solution:**
1. Run Command Prompt as Administrator
2. Or move project to non-protected folder (not Program Files)

---

### Antivirus blocking

**Symptoms:**
- Files disappear after creation
- Random permission errors
- Scripts fail unexpectedly

**Solution:**
1. Add project folder to antivirus exclusions
2. Temporarily disable antivirus to test
3. If it works, create permanent exclusion

---

## Browser Issues

### Dashboard shows blank page

**Solution:**
1. Clear browser cache (Ctrl+Shift+Delete)
2. Try different browser (Chrome, Firefox, Edge)
3. Disable browser extensions
4. Try incognito/private mode

---

### Can't upload CSV file

**Checklist:**
1. File must be CSV format
2. File must have all required columns
3. Column names must match exactly
4. File size reasonable (<100MB)

---

## Testing Issues

### Tests fail

**Common causes:**

1. **Models not trained**
   ```cmd
   python train.py
   ```

2. **Dependencies outdated**
   ```cmd
   pip install --upgrade -r requirements.txt
   ```

3. **API already running**
   - Close API before running tests
   - Tests start their own test server

---

## Network Issues

### Can't access dashboard from other computers

**Problem:** Want to access Dashboard from another PC on network

**Solution:**
1. Find your IP address:
   ```cmd
   ipconfig
   ```
   Look for IPv4 Address (e.g., 192.168.1.100)

2. Dashboard runs on localhost by default

3. Access from other PC:
   ```
   http://YOUR_IP:8501        (Dashboard)
   ```

4. Allow through firewall when prompted

**Note:** On YOUR computer, still use `http://localhost:8501`

---

## Still Having Issues?

### Diagnostic Steps

1. **Check Python version:**
   ```cmd
   python --version
   ```
   Should be 3.10 or higher

2. **Check dependencies:**
   ```cmd
   pip list
   ```
   Compare with requirements.txt

3. **Check models exist:**
   ```cmd
   dir models
   ```
   Should show .pkl files

4. **Test basic functionality:**
   ```cmd
   python -c "import pandas, sklearn, streamlit"
   ```
   Should run without errors

5. **Check ports:**
   ```cmd
   netstat -ano | findstr :8501
   ```
   Port should be free or used by dashboard

---

## Getting Help

If none of these solutions work:

1. Note the exact error message
2. Note what you were doing when error occurred
3. Check if models are trained
4. Verify Python version
5. Try running from fresh virtual environment

## Quick Reset

To completely reset and start fresh:

```cmd
REM Stop all services
REM Close all Command Prompt windows

REM Delete generated files
rmdir /s /q venv
rmdir /s /q models
del .env

REM Reinstall
INSTALL.bat
```

This will give you a clean installation.
