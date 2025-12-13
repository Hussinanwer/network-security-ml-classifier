# START HERE

Welcome to the Network Traffic Classification System! 👋

This file will guide you to the right documentation based on what you want to do.

## 🚀 I'm New - Just Want to Get Started

**Windows Users:**
1. Double-click `INSTALL.bat`
2. Wait 5-10 minutes
3. Double-click `start.bat`
4. Done!

**Linux/Mac Users:**
- See [QUICKSTART.md](QUICKSTART.md)

---

## 📖 What Do You Want to Do?

### Setup & Installation

| I want to... | Go to... |
|--------------|----------|
| Install on Windows (first time) | Double-click `INSTALL.bat` |
| Install on Linux/Mac | [QUICKSTART.md](QUICKSTART.md) |
| Understand the setup process | [WINDOWS_SETUP.md](WINDOWS_SETUP.md) |
| Configure settings | Edit `.env` file |

### Using the System

| I want to... | Go to... |
|--------------|----------|
| Use the web dashboard | Double-click `start.bat` or `run_dashboard.bat` |
| Make predictions | Open dashboard → http://localhost:8501 |
| Upload CSV for batch predictions | Dashboard → "Batch Prediction" tab |
| Check model performance | Dashboard → "Model Info" tab |

### Development

| I want to... | Go to... |
|--------------|----------|
| Understand the code | [CLAUDE.md](CLAUDE.md) |
| Train models | Double-click `retrain.bat` |
| Run tests | Double-click `run_tests.bat` |
| Understand project structure | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| Use models in Python code | See "Programmatic Usage" in [README.md](README.md) |

### Help & Troubleshooting

| I have a problem with... | Go to... |
|--------------------------|----------|
| Installation errors | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → Installation Issues |
| Dashboard not loading | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → Dashboard Issues |
| Batch files not working | [BATCH_FILES_README.txt](BATCH_FILES_README.txt) |
| General questions | [README.md](README.md) |

---

## 📚 Complete Documentation Index

### For Everyone

1. **[README.md](README.md)** - Complete project documentation
   - Features, installation, usage, configuration
   - Recommended first read after installation

2. **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
   - Fast setup guide
   - Basic usage examples

3. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Problem solving
   - Common issues and solutions
   - Error explanations
   - Quick fixes

### For Windows Users

4. **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** - Detailed Windows guide
   - Prerequisites
   - Step-by-step installation
   - Windows-specific tips

5. **[BATCH_FILES_README.txt](BATCH_FILES_README.txt)** - Batch file reference
   - What each .bat file does
   - When to use each one
   - Typical workflows


### For Developers

7. **[CLAUDE.md](CLAUDE.md)** - Development guide
   - Code architecture
   - Data processing pipeline
   - Development workflow

8. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project organization
   - File and folder structure
   - File relationships
   - Module dependencies

---

## 🎯 Quick Access

### Most Common Tasks

**First Time Setup (Windows):**
```
INSTALL.bat
```

**Daily Use:**
```
start.bat → Choose what to run
```

**Retrain Models:**
```
retrain.bat
```

**Run Tests:**
```
run_tests.bat
```

### URL to Bookmark

- **Dashboard:** http://localhost:8501

---

## 🗂️ Project Overview

This is a **machine learning system** that classifies network traffic into three categories:

1. **Normal SSH Traffic** (Class 0)
2. **FTP Traffic** (Class 1)
3. **Malicious/Attack Traffic** (Class 2)

### What's Included

✅ **3 Machine Learning Models**
- Random Forest (primary, 100% accuracy)
- Logistic Regression (87.5% accuracy)
- SVM (72.3% accuracy)

✅ **Interactive Web Dashboard**
- Easy-to-use Streamlit interface
- Single predictions with manual input
- Batch predictions from CSV files
- Real-time results and visualizations

✅ **Windows Batch Files**
- One-click installation
- Simplified menu
- Easy operation

✅ **Complete Documentation**
- Setup guides
- Usage examples
- Troubleshooting

---

## 🎓 Learning Path

**Beginner:**
1. Run `INSTALL.bat` (Windows) or follow `QUICKSTART.md`
2. Try the dashboard: `run_dashboard.bat`
3. Read `README.md` for overview
4. Experiment with predictions

**Intermediate:**
5. Understand project: `PROJECT_STRUCTURE.md`
6. Use models in Python: See README "Programmatic Usage"
7. Study code: `CLAUDE.md`

**Advanced:**
8. Modify preprocessing: `preprocessing.py`
9. Retrain models: `retrain.bat`
10. Run tests: `run_tests.bat`
11. Customize dashboard: `dashboard.py`

---

## 📞 Need Help?

**Check these in order:**

1. ✅ **Error Messages** - Read what the error says
2. ✅ **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common solutions
3. ✅ **[README.md](README.md)** - General information
4. ✅ **Specific Guides** - Windows, API, etc.
5. ✅ **Code Comments** - Look at the Python files

---

## 🌟 Next Steps

After reading this file:

**New User?** → Run `INSTALL.bat` (Windows) or see `QUICKSTART.md`

**Installed Already?** → Run `start.bat` (Windows) or `run_api.bat`/`run_dashboard.bat`

**Want to Learn More?** → Read `README.md`

**Having Issues?** → Check `TROUBLESHOOTING.md`

---

## 📋 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INSTALL:    INSTALL.bat (Windows) or see QUICKSTART.md    │
│                                                             │
│  RUN:        start.bat or run_dashboard.bat                │
│                                                             │
│  DASHBOARD:  http://localhost:8501                         │
│                                                             │
│  HELP:       TROUBLESHOOTING.md                            │
│                                                             │
│  RETRAIN:    retrain.bat                                   │
│                                                             │
│  TESTS:      run_tests.bat                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**Ready?** Pick your next step from the sections above! 🚀
