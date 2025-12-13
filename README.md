# Network Security ML Classifier

A machine learning system for classifying network traffic into three categories: Normal SSH Traffic, FTP Traffic, and Malicious/Attack Traffic. Achieves 100% test accuracy with Random Forest classifier.

> 🚀 **Quick Start:** See [SIMPLE_GUIDE.md](SIMPLE_GUIDE.md) for 3-step setup!

## 📋 Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Team Setup Guide](#team-setup-guide)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)

## ✨ Features

- **High Accuracy Models**: Random Forest (100%), Logistic Regression (87.5%), SVM (72.3%)
- **Interactive Web Dashboard**: Streamlit-based visual interface
- **Batch Processing**: Upload CSV files for bulk predictions
- **Complete Pipeline**: Automated preprocessing and feature engineering
- **Windows Support**: One-click batch files for easy operation
- **Well Documented**: Comprehensive guides for all skill levels

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning)

### Installation (Windows - Easiest)

1. **Clone the repository**
   ```cmd
   git clone https://github.com/hussinanwer/network-security-ml-classifier.git
   cd network-security-ml-classifier
   ```

2. **Run the installer**
   ```cmd
   INSTALL.bat
   ```
   This will:
   - Create virtual environment
   - Install all dependencies
   - Train the models (1-2 minutes)

3. **Start the dashboard**
   ```cmd
   start.bat
   ```
   Choose option [2] to launch the web dashboard

4. **Access the application**
   Open your browser to: http://localhost:8501

### Installation (Linux/Mac)

1. **Clone the repository**
   ```bash
   git clone https://github.com/hussinanwer/network-security-ml-classifier.git
   cd network-security-ml-classifier
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train models**
   ```bash
   python train.py
   ```

5. **Run dashboard**
   ```bash
   streamlit run dashboard.py
   ```

## 👥 Team Setup Guide

### First Time Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/hussinanwer/network-security-ml-classifier.git
   cd network-security-ml-classifier
   ```

2. **Install dependencies**
   - **Windows**: Double-click `INSTALL.bat`
   - **Linux/Mac**: Run the commands in "Installation (Linux/Mac)" above

3. **Verify installation**
   ```bash
   python -c "import pandas, sklearn, streamlit; print('All dependencies installed!')"
   ```

### Daily Workflow

1. **Pull latest changes**
   ```bash
   git pull origin main
   ```

2. **Activate virtual environment**
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

3. **Update dependencies (if requirements.txt changed)**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start working**
   - Run dashboard: `streamlit run dashboard.py` (or `run_dashboard.bat` on Windows)
   - Train models: `python train.py` (or `retrain.bat` on Windows)
   - Run tests: `pytest tests/` (or `run_tests.bat` on Windows)

### Branch Strategy

- `main` - Stable, working code
- `develop` - Integration branch
- `feature/*` - New features
- `bugfix/*` - Bug fixes

Example workflow:
```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes, commit
git add .
git commit -m "Add new neural network model"

# Push to GitHub
git push origin feature/new-model

# Create Pull Request on GitHub
```

## 📊 Usage

### Web Dashboard

The easiest way to use the system:

1. **Start dashboard**
   - Windows: `run_dashboard.bat` or `start.bat`
   - Linux/Mac: `streamlit run dashboard.py`

2. **Use features**
   - **Single Prediction**: Fill form → Get instant classification
   - **Batch Prediction**: Upload CSV → Get bulk results
   - **Model Info**: View performance metrics and confusion matrix

### Programmatic Usage

```python
import pickle
import pandas as pd

# Load models
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('models/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare data
data = {
    'src_ip': '192.168.113.129',
    'dst_ip': '192.168.113.130',
    'src_port': 44017,
    'dst_port': 22,
    # ... (all 35 features)
}
df = pd.DataFrame([data])

# Predict
X = preprocessor.transform(df)
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print(f"Prediction: {prediction}")  # 0, 1, or 2
print(f"Confidence: {probabilities[prediction]:.2%}")
```

## 📁 Project Structure

```
network-security-ml-classifier/
├── dashboard.py              # Streamlit web application
├── train.py                  # Model training script
├── preprocessing.py          # Data preprocessing pipeline
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
│
├── models/                   # Trained models (included!)
│   ├── rf_model.pkl         # Random Forest (primary)
│   ├── lr_model.pkl         # Logistic Regression
│   ├── svm_model.pkl        # SVM
│   ├── preprocessor.pkl     # Complete preprocessor
│   └── ...
│
├── tests/                    # Test suite
│   ├── test_preprocessing.py
│   └── test_data/
│
├── Windows Batch Files       # Windows automation
│   ├── INSTALL.bat          # One-click installer
│   ├── start.bat            # Interactive menu
│   ├── run_dashboard.bat    # Launch dashboard
│   ├── retrain.bat          # Retrain models
│   └── run_tests.bat        # Run tests
│
└── Documentation
    ├── README.md            # This file
    ├── SIMPLE_GUIDE.md      # 3-step quick guide
    ├── START_HERE.md        # Navigation guide
    ├── QUICKSTART.md        # Detailed quick start
    ├── TROUBLESHOOTING.md   # Common issues
    └── ...
```

## 📚 Documentation

- **[SIMPLE_GUIDE.md](SIMPLE_GUIDE.md)** - Super simple 3-step guide
- **[START_HERE.md](START_HERE.md)** - Navigation and overview
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed quick start
- **[CLAUDE.md](CLAUDE.md)** - Development guide and architecture
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common problems and solutions
- **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** - Windows-specific instructions
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project organization

## 🧪 Running Tests

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Windows users can use:
run_tests.bat
```

## 🔧 Development

### Retraining Models

After modifying preprocessing or adding new data:

```bash
python train.py
```

Or on Windows:
```cmd
retrain.bat
```

This will:
- Backup existing models to `models/backup/`
- Train all three models from scratch
- Save new models
- Display performance metrics

### Adding New Features

1. Create feature branch
2. Modify code (preprocessing, models, dashboard)
3. Update tests
4. Run tests: `pytest tests/ -v`
5. Commit and push
6. Create Pull Request

## 🎯 Model Performance

| Model | Test Accuracy | Precision | Recall | F1 Score |
|-------|---------------|-----------|--------|----------|
| **Random Forest** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| Logistic Regression | 87.5% | 87.8% | 87.5% | 87.5% |
| SVM | 72.3% | 74.8% | 72.3% | 72.1% |

**Primary Model:** Random Forest (100 trees)

## 📊 Dataset

- **File**: `network_traffic_multiclass_dataset.csv`
- **Size**: 2,073 samples with 36 features
- **Classes**:
  - Class 0: Normal SSH Traffic
  - Class 1: FTP Traffic
  - Class 2: Malicious/Attack Traffic
- **Features**: Network packet characteristics (IPs, ports, protocols, packet sizes, timing, TCP flags, flow metrics)

## 🛠️ Tech Stack

- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Web Interface**: Streamlit
- **Visualization**: plotly, matplotlib, seaborn
- **Class Balancing**: imbalanced-learn (SMOTE)
- **Testing**: pytest

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Make changes** and test thoroughly
4. **Commit** (`git commit -m 'Add AmazingFeature'`)
5. **Push** (`git push origin feature/AmazingFeature`)
6. **Open Pull Request**

### Contribution Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation
- Follow existing code style
- Run tests before submitting PR

## 📝 License

This project is part of a Network Security course.

## 👨‍💻 Authors

- **Hussain Anwer** - [@hussinanwer](https://github.com/hussinanwer)

## 🆘 Support

Having issues? Check these resources:

1. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common problems
2. **[Issues](https://github.com/hussinanwer/network-security-ml-classifier/issues)** - Report bugs or request features
3. **Documentation** - See files listed above

## 📞 Contact

- GitHub: [@hussinanwer](https://github.com/hussinanwer)
- Project: [network-security-ml-classifier](https://github.com/hussinanwer/network-security-ml-classifier)

## 🙏 Acknowledgments

- Dataset: Network Traffic Multiclass Dataset
- Frameworks: scikit-learn, Streamlit, pandas
- Course: Network Security

---

**⭐ Star this repo if you find it helpful!**

**🔗 Quick Links:**
- [Installation](#quick-start)
- [Team Setup](#team-setup-guide)
- [Documentation](#documentation)
- [Issues](https://github.com/hussinanwer/network-security-ml-classifier/issues)
