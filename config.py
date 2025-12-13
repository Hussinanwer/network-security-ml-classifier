"""
Configuration settings for the network traffic classification application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
MODELS_DIR = BASE_DIR / "models"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
LR_MODEL_PATH = MODELS_DIR / "lr_model.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.pkl"

# Data paths
DATA_DIR = BASE_DIR
DATASET_PATH = DATA_DIR / "network_traffic_multiclass_dataset.csv"

# Dashboard settings
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))

# Model settings
MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")  # Options: random_forest, logistic_regression, svm
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Training settings
SMOTE_STRATEGY = "auto"
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "100"))
RF_N_JOBS = int(os.getenv("RF_N_JOBS", "-1"))

# Feature engineering settings
ZERO_VARIANCE_THRESHOLD = 0
WEAK_CORRELATION_THRESHOLD = 0.05
HIGH_CORRELATION_THRESHOLD = 0.95

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Class labels
CLASS_LABELS = {
    0: "Normal SSH Traffic",
    1: "FTP Traffic",
    2: "Malicious/Attack Traffic"
}

# Final features (after preprocessing)
FINAL_FEATURES = [
    'src_ip',
    'src_port',
    'syn_count',
    'fin_count',
    'packets_per_second',
    'bytes_per_second',
    'forward_packets',
    'backward_bytes',
    'forward_backward_ratio',
    'avg_iat',
    'std_iat',
    'max_iat',
    'is_port_22',
    'is_ftp_port'
]

# Original features (before preprocessing)
ORIGINAL_FEATURES = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
    'duration', 'total_packets', 'total_bytes', 'min_packet_size',
    'max_packet_size', 'avg_packet_size', 'std_packet_size',
    'syn_count', 'ack_count', 'fin_count', 'rst_count',
    'psh_count', 'urg_count', 'packets_per_second',
    'bytes_per_second', 'bytes_per_packet', 'forward_packets',
    'backward_packets', 'forward_bytes', 'backward_bytes',
    'forward_backward_ratio', 'avg_iat', 'std_iat',
    'min_iat', 'max_iat', 'syn_ack_ratio',
    'is_port_22', 'is_port_6200', 'is_ftp_port', 'is_ftp_data_port'
]
