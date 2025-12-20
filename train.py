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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif,RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
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
    Train multiple models matching the notebook configuration.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary containing all trained models and their metrics
    """
    models = {}

    # 1. Random Forest (with regularization from notebook)
    print("\n" + "="*70)
    print("Training Random Forest Classifier...")
    print("="*70)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
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

    models['random_forest'] = rf_metrics

    # 2. SVM (matching notebook parameters)
    print("\n" + "="*70)
    print("Training SVM...")
    print("="*70)

    svm_model = svm.SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,  # Enable probability estimates for predict_proba()
        random_state=42
    )
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

    # 3. Decision Tree (with regularization from notebook)
    print("\n" + "="*70)
    print("Training Decision Tree...")
    print("="*70)

    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    y_train_pred_dt = dt_model.predict(X_train)
    y_test_pred_dt = dt_model.predict(X_test)

    dt_metrics = {
        'name': 'Decision Tree',
        'model': dt_model,
        'train_accuracy': accuracy_score(y_train, y_train_pred_dt),
        'test_accuracy': accuracy_score(y_test, y_test_pred_dt),
        'precision': precision_score(y_test, y_test_pred_dt, average='weighted'),
        'recall': recall_score(y_test, y_test_pred_dt, average='weighted'),
        'f1_score': f1_score(y_test, y_test_pred_dt, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred_dt),
        'classification_report': classification_report(y_test, y_test_pred_dt)
    }

    print(f"Test Accuracy: {dt_metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {dt_metrics['f1_score']:.4f}")

    models['decision_tree'] = dt_metrics

    # 4. Naive Bayes
    print("\n" + "="*70)
    print("Training Naive Bayes...")
    print("="*70)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    y_train_pred_nb = nb_model.predict(X_train)
    y_test_pred_nb = nb_model.predict(X_test)

    nb_metrics = {
        'name': 'Naive Bayes',
        'model': nb_model,
        'train_accuracy': accuracy_score(y_train, y_train_pred_nb),
        'test_accuracy': accuracy_score(y_test, y_test_pred_nb),
        'precision': precision_score(y_test, y_test_pred_nb, average='weighted'),
        'recall': recall_score(y_test, y_test_pred_nb, average='weighted'),
        'f1_score': f1_score(y_test, y_test_pred_nb, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred_nb),
        'classification_report': classification_report(y_test, y_test_pred_nb)
    }

    print(f"Test Accuracy: {nb_metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {nb_metrics['f1_score']:.4f}")

    models['naive_bayes'] = nb_metrics

    # 5. Logistic Regression (matching notebook parameters)
    print("\n" + "="*70)
    print("Training Logistic Regression...")
    print("="*70)

    lr_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
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

    return models


def save_models(preprocessor, models, selector, models_dir='models'):
    """
    Save preprocessor, models, and feature selector to disk.

    Args:
        preprocessor: Fitted NetworkTrafficPreprocessor
        models: Dictionary of trained models
        selector: Fitted SelectKBest feature selector
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

    print(f"Saving label encoders to {models_dir}/label_encoders.pkl")
    with open(f'{models_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(preprocessor.label_encoders, f)

    print(f"Saving preprocessor to {models_dir}/preprocessor.pkl")
    with open(f'{models_dir}/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save feature selector
    print(f"Saving feature selector to {models_dir}/feature_selector.pkl")
    with open(f'{models_dir}/feature_selector.pkl', 'wb') as f:
        pickle.dump(selector, f)

    # Save all 5 models
    print(f"Saving Random Forest model to {models_dir}/rf_model.pkl")
    with open(f'{models_dir}/rf_model.pkl', 'wb') as f:
        pickle.dump(models['random_forest']['model'], f)

    print(f"Saving SVM model to {models_dir}/svm_model.pkl")
    with open(f'{models_dir}/svm_model.pkl', 'wb') as f:
        pickle.dump(models['svm']['model'], f)

    print(f"Saving Decision Tree model to {models_dir}/dt_model.pkl")
    with open(f'{models_dir}/dt_model.pkl', 'wb') as f:
        pickle.dump(models['decision_tree']['model'], f)

    print(f"Saving Naive Bayes model to {models_dir}/nb_model.pkl")
    with open(f'{models_dir}/nb_model.pkl', 'wb') as f:
        pickle.dump(models['naive_bayes']['model'], f)

    print(f"Saving Logistic Regression model to {models_dir}/lr_model.pkl")
    with open(f'{models_dir}/lr_model.pkl', 'wb') as f:
        pickle.dump(models['logistic_regression']['model'], f)

    # Save model metadata
    metadata = {
        'random_forest': {
            'test_accuracy': models['random_forest']['test_accuracy'],
            'precision': models['random_forest']['precision'],
            'recall': models['random_forest']['recall'],
            'f1_score': models['random_forest']['f1_score'],
            'confusion_matrix': models['random_forest']['confusion_matrix'].tolist(),
        },
        'svm': {
            'test_accuracy': models['svm']['test_accuracy'],
            'precision': models['svm']['precision'],
            'recall': models['svm']['recall'],
            'f1_score': models['svm']['f1_score'],
            'confusion_matrix': models['svm']['confusion_matrix'].tolist(),
        },
        'decision_tree': {
            'test_accuracy': models['decision_tree']['test_accuracy'],
            'precision': models['decision_tree']['precision'],
            'recall': models['decision_tree']['recall'],
            'f1_score': models['decision_tree']['f1_score'],
        },
        'naive_bayes': {
            'test_accuracy': models['naive_bayes']['test_accuracy'],
            'precision': models['naive_bayes']['precision'],
            'recall': models['naive_bayes']['recall'],
            'f1_score': models['naive_bayes']['f1_score'],
        },
        'logistic_regression': {
            'test_accuracy': models['logistic_regression']['test_accuracy'],
            'precision': models['logistic_regression']['precision'],
            'recall': models['logistic_regression']['recall'],
            'f1_score': models['logistic_regression']['f1_score'],
        },
        'final_features': preprocessor.final_features,
        'removed_features': preprocessor.removed_features
    }

    print(f"Saving metadata to {models_dir}/model_metadata.pkl")
    with open(f'{models_dir}/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("\nAll models saved successfully!")


def main():
    """Main training pipeline matching the notebook."""
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

    print("\nLabel distribution in training set:")
    print(y_train.value_counts().sort_index())

    # Scale features
    print("\nScaling features...")
    X_train_scaled = preprocessor.scaler.transform(X_train)
    X_test_scaled = preprocessor.scaler.transform(X_test)

    # RFE Feature Selection (Recursive Feature Elimination)
    print("\n" + "="*70)
    print("FEATURE SELECTION - RFE (Recursive Feature Elimination)")
    print("="*70)
    k_best = min(15, X_train_scaled.shape[1])  # Use all features if less than 15
    print(f"Selecting top {k_best} features using RFE with Random Forest (out of {X_train_scaled.shape[1]} available)...")

    # Use Random Forest as estimator for RFE
    rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    selector = RFE(estimator=rf_estimator, n_features_to_select=k_best, step=1)

    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Get selected feature names
    selected_features = [preprocessor.final_features[i] for i in range(len(preprocessor.final_features))
                        if selector.get_support()[i]]

    print(f"\nSelected {k_best} features:")
    for i, feature in enumerate(selected_features, 1):
        feature_idx = preprocessor.final_features.index(feature)
        ranking = selector.ranking_[feature_idx]
        print(f"   {i:2d}. {feature:30s} (Rank: {ranking})")

    print(f"\nRFE feature selection complete!")

    # Train models on selected features
    models = train_and_evaluate_models(
        X_train_selected, y_train,
        X_test_selected, y_test
    )

    # Save everything
    save_models(preprocessor, models, selector)

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