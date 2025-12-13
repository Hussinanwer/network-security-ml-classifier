# Project Structure

Complete overview of the Network Traffic Classification project organization.

## Directory Layout

```
network-traffic-classification/
│
├── models/                       # Trained models (generated)
│   ├── backup/                   # Backup of previous models
│   ├── rf_model.pkl             # Random Forest model (primary)
│   ├── lr_model.pkl             # Logistic Regression model
│   ├── svm_model.pkl            # SVM model
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoder.pkl        # Protocol encoder
│   ├── preprocessor.pkl         # Complete preprocessor
│   └── model_metadata.pkl       # Model metrics
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_preprocessing.py    # Preprocessing unit tests
│   ├── test_api.py              # API integration tests
│   └── test_data/               # Test data files
│
├── venv/                         # Virtual environment (generated)
│
├── Core Python Files
├── preprocessing.py              # Preprocessing pipeline
├── train.py                      # Model training script
├── dashboard.py                  # Streamlit web interface
├── config.py                     # Configuration settings
├── run_api.py                    # API launcher script
├── run_dashboard.py              # Dashboard launcher script
│
├── Windows Batch Files
├── INSTALL.bat                   # One-click installer
├── start.bat                     # Interactive menu
├── setup.bat                     # Setup and training
├── run_dashboard.bat             # Run dashboard
├── retrain.bat                   # Retrain models
├── run_tests.bat                 # Run test suite
│
├── Configuration Files
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .env                          # User environment (generated)
├── config.py                     # Application config
├── pytest.ini                    # Pytest configuration
├── .gitignore                    # Git ignore rules
│
├── Documentation
├── README.md                     # Main documentation
├── SIMPLE_GUIDE.md               # Super simple guide
├── QUICKSTART.md                 # Quick start guide
├── START_HERE.md                 # Navigation guide
├── CLAUDE.md                     # Claude Code guidance
├── WINDOWS_SETUP.md              # Windows setup guide
├── BATCH_FILES_README.txt        # Batch files guide
├── TROUBLESHOOTING.md            # Common issues & solutions
├── PROJECT_STRUCTURE.md          # This file
├── CHANGELOG.md                  # Version history
│
├── Data Files
├── network_traffic_multiclass_dataset.csv   # Training dataset
└── Network_Security_project (2).ipynb        # Original notebook

```

## File Descriptions

### Core Modules

| File | Purpose | Used By |
|------|---------|---------|
| `preprocessing.py` | Preprocessing pipeline implementation | train.py, api, dashboard |
| `train.py` | Model training script | setup.bat, retrain.bat |
| `dashboard.py` | Streamlit web interface | run_dashboard.bat |
| `config.py` | Application settings | All Python scripts |

### Scripts

| File | Purpose |
|------|---------|
| `run_dashboard.py` | Launch Streamlit dashboard |

### Batch Files (Windows)

| File | Purpose | When to Use |
|------|---------|-------------|
| `INSTALL.bat` | Complete installation wizard | First time setup |
| `start.bat` | Interactive menu | Daily use |
| `setup.bat` | Environment setup | First time or reset |
| `run_dashboard.bat` | Start dashboard | Use the web interface |
| `retrain.bat` | Retrain models | After data changes |
| `run_tests.bat` | Run tests | Verify functionality |

### Documentation

| File | Audience | Content |
|------|----------|---------|
| `README.md` | All users | Complete documentation |
| `SIMPLE_GUIDE.md` | Beginners | Super simple 3-step guide |
| `QUICKSTART.md` | New users | 5-minute setup guide |
| `START_HERE.md` | All users | Navigation guide |
| `WINDOWS_SETUP.md` | Windows users | Detailed Windows guide |
| `BATCH_FILES_README.txt` | Windows users | Batch file reference |
| `TROUBLESHOOTING.md` | All users | Problem solving |
| `CLAUDE.md` | Developers | Development guide |
| `PROJECT_STRUCTURE.md` | All users | This file |
| `CHANGELOG.md` | All users | Version history |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `.env.example` | Environment variable template |
| `.env` | User-specific settings (generated) |
| `pytest.ini` | Test runner configuration |
| `.gitignore` | Git version control exclusions |


## Generated Directories

These directories are created automatically:

| Directory | Created By | Purpose |
|-----------|------------|---------|
| `venv/` | setup.bat | Python virtual environment |
| `models/` | train.py | Saved machine learning models |
| `models/backup/` | retrain.bat | Previous model versions |
| `__pycache__/` | Python | Compiled bytecode (ignored) |
| `.pytest_cache/` | pytest | Test cache (ignored) |

## Data Flow

```
Raw CSV Data
    ↓
preprocessing.py (NetworkTrafficPreprocessor)
    ↓
train.py (Model Training)
    ↓
models/ (Saved Models)
    ↓
┌────────────┬────────────┐
│            │            │
api/app.py   dashboard.py (Production Use)
│            │
REST API     Web UI
```

## Development Workflow

```
1. Setup
   INSTALL.bat → setup.bat → train.py → models/

2. Development
   Edit code → run_tests.bat → Verify changes

3. Deployment
   ├─ Local: run_api.bat / run_dashboard.bat
   └─ Docker: docker-compose up

4. Updates
   retrain.bat → Backup old models → Train new models
```

## File Relationships

### Model Training Pipeline
```
network_traffic_multiclass_dataset.csv
    ↓
train.py
    ├─ imports preprocessing.py
    ├─ imports config.py
    └─ outputs to models/
```

### Dashboard Service
```
run_dashboard.bat
    ↓
run_dashboard.py
    ↓
dashboard.py
    ├─ imports preprocessing.py
    └─ loads models/
```

## Important Notes

### Do NOT Delete
- `network_traffic_multiclass_dataset.csv` - Required for training
- `models/` directory - Contains trained models
- `preprocessing.py` - Core preprocessing logic
- `.env` - Your configuration settings

### Safe to Delete (Will be regenerated)
- `venv/` - Can recreate with setup.bat
- `models/backup/` - Old model backups
- `__pycache__/` - Python cache
- `.pytest_cache/` - Test cache

### Version Control
Files in `.gitignore`:
- `venv/` - Virtual environment
- `.env` - Personal settings
- `__pycache__/` - Python cache
- `.pytest_cache/` - Test cache
- `.ipynb_checkpoints/` - Jupyter checkpoints

Files to track:
- All `.py` files
- All `.bat` files
- All `.md` files
- `requirements.txt`
- `.env.example`
- Docker files

## Module Dependencies

```
config.py (no dependencies)
    ↓
preprocessing.py
    ├─ Uses: numpy, pandas, sklearn
    └─ Imported by: train.py, api/app.py, dashboard.py
    ↓
train.py
    ├─ Uses: preprocessing.py, config.py, sklearn, imblearn
    └─ Outputs: models/*.pkl
    ↓
api/app.py & dashboard.py
    ├─ Uses: preprocessing.py, config.py, models/*.pkl
    └─ Serve predictions
```

## Port Usage

| Service | Default Port | Configurable In |
|---------|--------------|-----------------|
| Dashboard | 8501 | `.env` (DASHBOARD_PORT) |

## Storage Requirements

| Item | Size (Approx) |
|------|---------------|
| Dataset | 630 KB |
| Models | < 200 KB |
| Virtual Environment | 50-200 MB |
| Dependencies | 200-500 MB |
| **Total** | ~300-700 MB |

## Adding New Files

### New Python Module
```
1. Create file: new_module.py
2. Add imports to: api/app.py or dashboard.py
3. Add tests: tests/test_new_module.py
4. Run: run_tests.bat
```

### New Documentation
```
1. Create: NEW_DOC.md
2. Add link in: README.md
3. Update: PROJECT_STRUCTURE.md
```

### New Batch File
```
1. Create: new_task.bat
2. Add to: start.bat menu
3. Document in: BATCH_FILES_README.txt
```

## Quick Navigation

**For new users:** Start with `README.md` or `QUICKSTART.md`
**For Windows users:** See `WINDOWS_SETUP.md`
**For API users:** See `API.md`
**For developers:** See `CLAUDE.md`
**For problems:** See `TROUBLESHOOTING.md`

## Maintenance

### Regular Tasks
- Retrain models: `retrain.bat` (backs up old models)
- Run tests: `run_tests.bat`
- Update dependencies: `pip install --upgrade -r requirements.txt`

### Cleanup
```cmd
REM Remove cache files
del /s /q __pycache__
rmdir /s /q .pytest_cache

REM Remove old models
del models\backup\*

REM Reset environment
rmdir /s /q venv
INSTALL.bat
```

## Support Files Location

| Need | See File |
|------|----------|
| How to install | README.md, WINDOWS_SETUP.md |
| How to use | QUICKSTART.md |
| API reference | API.md |
| Troubleshooting | TROUBLESHOOTING.md |
| Batch file help | BATCH_FILES_README.txt |
| Project overview | This file |
