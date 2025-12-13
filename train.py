"""
Training script for network traffic classification models.
This script trains models from scratch and saves them to the models/ directory.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from preprocessing import NetworkTrafficPreprocessor
import warnings
warnings.filterwarnings('ignore')


def load_data(csv_path='network_traffic_multiclass_dataset.csv'):
    """Load the network traffic dataset."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    return df


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and return the best one.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary containing all trained models and their metrics
    """
    models = {}

    # 1. Random Forest (Primary model - best performance in notebook)
    print("\n" + "="*70)
    print("Training Random Forest Classifier...")
    print("="*70)

    rf_model = RandomForestClassifier(
        n_estimators=100,  # Increased from 3 for production
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)

    rf_metrics = {
        'name': 'Random Forest',
        'model': rf_model,
        'train_accuracy': accuracy_score(y_train, y_train_pred_rf),
        'test_accuracy': accuracy_score(y_test, y_test_pred_rf),
        'precision': precision_score(y_test, y_test_pred_rf, average='weighted'),
        'recall': recall_score(y_test, y_test_pred_rf, average='weighted'),
        'f1_score': f1_score(y_test, y_test_pred_rf, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred_rf),
        'classification_report': classification_report(y_test, y_test_pred_rf)
    }

    print(f"Train Accuracy: {rf_metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {rf_metrics['test_accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall: {rf_metrics['recall']:.4f}")
    print(f"F1 Score: {rf_metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(rf_metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(rf_metrics['classification_report'])

    models['random_forest'] = rf_metrics

    # 2. Logistic Regression
    print("\n" + "="*70)
    print("Training Logistic Regression...")
    print("="*70)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    y_train_pred_lr = lr_model.predict(X_train)
    y_test_pred_lr = lr_model.predict(X_test)

    lr_metrics = {
        'name': 'Logistic Regression',
        'model': lr_model,
        'train_accuracy': accuracy_score(y_train, y_train_pred_lr),
        'test_accuracy': accuracy_score(y_test, y_test_pred_lr),
        'precision': precision_score(y_test, y_test_pred_lr, average='weighted'),
        'recall': recall_score(y_test, y_test_pred_lr, average='weighted'),
        'f1_score': f1_score(y_test, y_test_pred_lr, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred_lr),
        'classification_report': classification_report(y_test, y_test_pred_lr)
    }

    print(f"Test Accuracy: {lr_metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {lr_metrics['f1_score']:.4f}")

    models['logistic_regression'] = lr_metrics

    # 3. SVM
    print("\n" + "="*70)
    print("Training SVM...")
    print("="*70)

    svm_model = svm.SVC(kernel='rbf', C=4, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)

    y_train_pred_svm = svm_model.predict(X_train)
    y_test_pred_svm = svm_model.predict(X_test)

    svm_metrics = {
        'name': 'SVM',
        'model': svm_model,
        'train_accuracy': accuracy_score(y_train, y_train_pred_svm),
        'test_accuracy': accuracy_score(y_test, y_test_pred_svm),
        'precision': precision_score(y_test, y_test_pred_svm, average='weighted'),
        'recall': recall_score(y_test, y_test_pred_svm, average='weighted'),
        'f1_score': f1_score(y_test, y_test_pred_svm, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred_svm),
        'classification_report': classification_report(y_test, y_test_pred_svm)
    }

    print(f"Test Accuracy: {svm_metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {svm_metrics['f1_score']:.4f}")

    models['svm'] = svm_metrics

    return models


def save_models(preprocessor, models, models_dir='models'):
    """
    Save preprocessor and models to disk.

    Args:
        preprocessor: Fitted NetworkTrafficPreprocessor
        models: Dictionary of trained models
        models_dir: Directory to save models
    """
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Saving models...")
    print("="*70)

    # Save preprocessor components
    print(f"Saving scaler to {models_dir}/scaler.pkl")
    with open(f'{models_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(preprocessor.scaler, f)

    print(f"Saving label encoder to {models_dir}/label_encoder.pkl")
    with open(f'{models_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(preprocessor.label_encoder, f)

    print(f"Saving preprocessor to {models_dir}/preprocessor.pkl")
    with open(f'{models_dir}/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save Random Forest (primary model)
    print(f"Saving Random Forest model to {models_dir}/rf_model.pkl")
    with open(f'{models_dir}/rf_model.pkl', 'wb') as f:
        pickle.dump(models['random_forest']['model'], f)

    # Save other models
    print(f"Saving Logistic Regression model to {models_dir}/lr_model.pkl")
    with open(f'{models_dir}/lr_model.pkl', 'wb') as f:
        pickle.dump(models['logistic_regression']['model'], f)

    print(f"Saving SVM model to {models_dir}/svm_model.pkl")
    with open(f'{models_dir}/svm_model.pkl', 'wb') as f:
        pickle.dump(models['svm']['model'], f)

    # Save model metadata
    metadata = {
        'random_forest': {
            'test_accuracy': models['random_forest']['test_accuracy'],
            'precision': models['random_forest']['precision'],
            'recall': models['random_forest']['recall'],
            'f1_score': models['random_forest']['f1_score'],
            'confusion_matrix': models['random_forest']['confusion_matrix'].tolist(),
        },
        'logistic_regression': {
            'test_accuracy': models['logistic_regression']['test_accuracy'],
            'precision': models['logistic_regression']['precision'],
            'recall': models['logistic_regression']['recall'],
            'f1_score': models['logistic_regression']['f1_score'],
        },
        'svm': {
            'test_accuracy': models['svm']['test_accuracy'],
            'precision': models['svm']['precision'],
            'recall': models['svm']['recall'],
            'f1_score': models['svm']['f1_score'],
        },
        'final_features': preprocessor.final_features,
        'removed_features': preprocessor.removed_features
    }

    print(f"Saving metadata to {models_dir}/model_metadata.pkl")
    with open(f'{models_dir}/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("\nAll models saved successfully!")


def main():
    """Main training pipeline."""
    print("="*70)
    print("NETWORK TRAFFIC CLASSIFICATION - MODEL TRAINING")
    print("="*70)

    # Load data
    df = load_data()

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = NetworkTrafficPreprocessor()

    # Fit preprocessor and get processed features
    print("\nPreprocessing data...")
    X, y = preprocessor.fit(df, target_column='label')

    print(f"Final features ({len(preprocessor.final_features)}): {preprocessor.final_features}")
    print(f"Removed features: {sum(len(v) for v in preprocessor.removed_features.values())} total")

    # Train-test split (80/20 with stratification)
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Apply SMOTE to training data only
    print("\nApplying SMOTE to balance training classes...")
    print("Before SMOTE:")
    print(y_train.value_counts().sort_index())

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts().sort_index())

    # Scale features
    print("\nScaling features...")
    X_train_scaled = preprocessor.scaler.transform(X_train_smote)
    X_test_scaled = preprocessor.scaler.transform(X_test)

    # Train models
    models = train_and_evaluate_models(
        X_train_scaled, y_train_smote,
        X_test_scaled, y_test
    )

    # Save everything
    save_models(preprocessor, models)

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Test Accuracy':<15} {'F1 Score':<15}")
    print("-"*70)
    for model_name, metrics in models.items():
        print(f"{metrics['name']:<25} {metrics['test_accuracy']:<15.4f} {metrics['f1_score']:<15.4f}")

    print("\n" + "="*70)
    print("Training complete! Models saved to models/ directory.")
    print("="*70)


if __name__ == '__main__':
    main()
