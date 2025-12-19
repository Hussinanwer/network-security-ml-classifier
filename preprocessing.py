"""
Preprocessing module for network traffic classification.
This module preserves the exact preprocessing pipeline from the notebook.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_raw_data(df, label_encoders=None, fit_encoders=False):
    """
    Preprocess raw network traffic data.
    This function handles encoding of categorical features (src_ip, dst_ip, protocol) using LabelEncoder.
    This EXACTLY matches the notebook preprocessing approach, with handling for unseen values.

    Args:
        df (pd.DataFrame): Raw input data
        label_encoders (dict, optional): Dict of pre-fitted label encoders
        fit_encoders (bool): Whether to fit new label encoders

    Returns:
        tuple: (processed_df, label_encoders_dict)
    """
    df = df.copy()

    if label_encoders is None:
        label_encoders = {}

    # Encode categorical features using LabelEncoder (src_ip, dst_ip, protocol)
    # This matches the notebook exactly!
    categorical_cols = ['src_ip', 'dst_ip', 'protocol']

    for col in categorical_cols:
        if col in df.columns:
            if fit_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            elif col in label_encoders:
                # Handle unseen values during transform
                # Map unseen values to -1 (will be handled by the model)
                le = label_encoders[col]

                # Get known classes
                known_classes = set(le.classes_)

                # For each value, encode if known, otherwise use -1
                def safe_encode(val):
                    val_str = str(val)
                    if val_str in known_classes:
                        return le.transform([val_str])[0]
                    else:
                        # Unseen value - encode as -1
                        # This will be treated as a new/unknown category
                        return -1

                df[col] = df[col].apply(safe_encode)
            else:
                # If no encoder and not fitting, just convert to string
                df[col] = df[col].astype(str)

    # Handle infinite values
    df = df.replace([np.inf, -np.inf], 0)

    # Fill missing values
    df = df.fillna(0)

    return df, label_encoders


def remove_leaky_features(X):
    """
    Remove features that are too perfect indicators (cause data leakage).
    Based on notebook analysis, is_port_6200 and is_ftp_data_port are leaky.
    This EXACTLY matches the notebook preprocessing.

    Args:
        X (pd.DataFrame): Feature dataframe

    Returns:
        tuple: (filtered_X, removed_features)
    """
    leaky_features = ['is_port_6200', 'is_ftp_data_port']
    existing_leaky = [f for f in leaky_features if f in X.columns]

    if len(existing_leaky) > 0:
        X = X.drop(columns=existing_leaky)

    return X, existing_leaky


def prepare_features_for_prediction(df, expected_features):
    """
    Prepare features for prediction by ensuring correct column order.

    Args:
        df (pd.DataFrame): Input dataframe with features
        expected_features (list): List of expected feature names in order

    Returns:
        pd.DataFrame: Dataframe with features in correct order
    """
    # Check for missing features
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Check for extra features
    extra_features = set(df.columns) - set(expected_features)
    if extra_features:
        print(f"Warning: Extra features will be ignored: {extra_features}")

    # Return features in correct order
    return df[expected_features]


class NetworkTrafficPreprocessor:
    """
    Complete preprocessing pipeline for network traffic classification.
    This class EXACTLY matches the notebook preprocessing pipeline:
    1. Encode categorical features (src_ip, dst_ip, protocol) with LabelEncoder
    2. Remove leaky features (is_port_6200, is_ftp_data_port)
    3. Scale features with StandardScaler

    Note: Feature selection (ANOVA SelectKBest) is done separately in train.py
    """

    def __init__(self):
        self.label_encoders = None
        self.scaler = None
        self.final_features = None
        self.removed_features = {
            'leaky': []
        }

    def fit(self, df, target_column='label'):
        """
        Fit the preprocessor on training data.
        EXACTLY matches the notebook preprocessing before feature selection.

        Args:
            df (pd.DataFrame): Training dataframe with all original features
            target_column (str): Name of target column

        Returns:
            tuple: (X_processed, y)
        """
        df = df.copy()

        # Separate target
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(target_column, axis=1)
        else:
            raise ValueError(f"Target column '{target_column}' not found")

        # Step 1: Encode categorical features (src_ip, dst_ip, protocol) using LabelEncoder
        # This matches the notebook exactly!
        X, self.label_encoders = preprocess_raw_data(X, fit_encoders=True)

        # Step 2: Remove leaky features ONLY (is_port_6200, is_ftp_data_port)
        # The notebook does NOT remove zero variance, weak correlation, or high correlation features!
        X, leaky = remove_leaky_features(X)
        self.removed_features['leaky'] = leaky

        # Store final features (after removing leaky features, before feature selection)
        self.final_features = X.columns.tolist()

        # Step 3: Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        return X, y

    def transform(self, df):
        """
        Transform new data using fitted preprocessor.
        EXACTLY matches the notebook preprocessing.

        Args:
            df (pd.DataFrame): Input dataframe with raw features

        Returns:
            np.ndarray: Scaled feature array ready for feature selection/prediction
        """
        df = df.copy()

        # Step 1: Encode categorical features using saved label encoders
        X, _ = preprocess_raw_data(df, label_encoders=self.label_encoders, fit_encoders=False)

        # Step 2: Remove leaky features
        for feat in self.removed_features['leaky']:
            if feat in X.columns:
                X = X.drop(columns=[feat])

        # Ensure correct feature order
        X = prepare_features_for_prediction(X, self.final_features)

        # Step 3: Scale
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit_transform(self, df, target_column='label'):
        """
        Fit and transform in one step.

        Args:
            df (pd.DataFrame): Training dataframe
            target_column (str): Name of target column

        Returns:
            tuple: (X_scaled, y)
        """
        X, y = self.fit(df, target_column)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y
