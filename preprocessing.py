"""
Preprocessing module for network traffic classification.
This module preserves the exact preprocessing pipeline from the notebook.
"""

import socket
import struct
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def ip_to_int(ip):
    """
    Convert IP address string to integer representation.

    Args:
        ip (str): IP address string (e.g., '192.168.1.1')

    Returns:
        int: Integer representation of IP address
    """
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except:
        return 0


def preprocess_raw_data(df, label_encoder=None, fit_encoder=False):
    """
    Preprocess raw network traffic data.
    This function handles IP encoding and protocol encoding.

    Args:
        df (pd.DataFrame): Raw input data
        label_encoder (LabelEncoder, optional): Pre-fitted label encoder for protocol
        fit_encoder (bool): Whether to fit a new label encoder

    Returns:
        tuple: (processed_df, label_encoder)
    """
    df = df.copy()

    # Convert IP addresses to integers
    if 'src_ip' in df.columns:
        df['src_ip'] = df['src_ip'].apply(ip_to_int)
    if 'dst_ip' in df.columns:
        df['dst_ip'] = df['dst_ip'].apply(ip_to_int)

    # Encode protocol
    if 'protocol' in df.columns:
        if fit_encoder:
            label_encoder = LabelEncoder()
            df['protocol'] = label_encoder.fit_transform(df['protocol'])
        elif label_encoder is not None:
            df['protocol'] = label_encoder.transform(df['protocol'])

    return df, label_encoder


def remove_zero_variance_features(X, variance_threshold=0):
    """
    Remove features with zero variance.

    Args:
        X (pd.DataFrame): Feature dataframe
        variance_threshold (float): Variance threshold

    Returns:
        tuple: (filtered_X, removed_features)
    """
    variances = X.var()
    zero_var_features = variances[variances == variance_threshold].index.tolist()

    if len(zero_var_features) > 0:
        X = X.drop(columns=zero_var_features)

    return X, zero_var_features


def remove_weak_features(X, y, correlation_threshold=0.05):
    """
    Remove features with weak correlation to target.

    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        correlation_threshold (float): Minimum absolute correlation

    Returns:
        tuple: (filtered_X, removed_features)
    """
    target_correlations = X.corrwith(y).abs()
    weak_features = target_correlations[target_correlations < correlation_threshold].index.tolist()

    if len(weak_features) > 0:
        X = X.drop(columns=weak_features)

    return X, weak_features


def remove_highly_correlated_features(X, y, correlation_threshold=0.95):
    """
    Remove highly correlated features, keeping ones with higher target correlation.

    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        correlation_threshold (float): Correlation threshold for feature pairs

    Returns:
        tuple: (filtered_X, removed_features)
    """
    corr_matrix = X.corr()
    target_correlations = X.corrwith(y).abs()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                high_corr_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    to_remove = set()
    for pair in high_corr_pairs:
        feat1 = pair['Feature1']
        feat2 = pair['Feature2']
        # Keep feature with higher target correlation
        if target_correlations[feat1] < target_correlations[feat2]:
            to_remove.add(feat1)
        else:
            to_remove.add(feat2)

    to_remove = list(to_remove)
    if len(to_remove) > 0:
        X = X.drop(columns=to_remove)

    return X, to_remove


def get_final_features():
    """
    Returns the list of final features after all preprocessing.
    These are the 14 features that should be used for prediction.

    Returns:
        list: List of feature names
    """
    return [
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


def prepare_features_for_prediction(df, expected_features=None):
    """
    Prepare features for prediction by ensuring correct column order.

    Args:
        df (pd.DataFrame): Input dataframe with features
        expected_features (list, optional): List of expected feature names in order

    Returns:
        pd.DataFrame: Dataframe with features in correct order
    """
    if expected_features is None:
        expected_features = get_final_features()

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
    This class encapsulates all preprocessing steps in the correct order.
    """

    def __init__(self):
        self.label_encoder = None
        self.scaler = None
        self.final_features = get_final_features()
        self.removed_features = {
            'zero_variance': [],
            'weak_correlation': [],
            'high_correlation': []
        }

    def fit(self, df, target_column='label'):
        """
        Fit the preprocessor on training data.

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

        # Step 1: IP and protocol encoding
        X, self.label_encoder = preprocess_raw_data(X, fit_encoder=True)

        # Step 2: Remove zero variance features
        X, zero_var = remove_zero_variance_features(X)
        self.removed_features['zero_variance'] = zero_var

        # Step 3: Remove weak features
        X, weak = remove_weak_features(X, y, correlation_threshold=0.05)
        self.removed_features['weak_correlation'] = weak

        # Step 4: Remove highly correlated features
        X, high_corr = remove_highly_correlated_features(X, y, correlation_threshold=0.95)
        self.removed_features['high_correlation'] = high_corr

        # Step 5: Ensure we have the expected final features
        X = prepare_features_for_prediction(X, self.final_features)

        # Step 6: Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        return X, y

    def transform(self, df):
        """
        Transform new data using fitted preprocessor.

        Args:
            df (pd.DataFrame): Input dataframe with raw features

        Returns:
            np.ndarray: Scaled feature array ready for prediction
        """
        df = df.copy()

        # Step 1: IP and protocol encoding
        X, _ = preprocess_raw_data(df, label_encoder=self.label_encoder, fit_encoder=False)

        # Step 2-4: Remove features (using saved list from fit)
        features_to_remove = (
            self.removed_features['zero_variance'] +
            self.removed_features['weak_correlation'] +
            self.removed_features['high_correlation']
        )

        for feat in features_to_remove:
            if feat in X.columns:
                X = X.drop(columns=[feat])

        # Step 5: Ensure correct feature order
        X = prepare_features_for_prediction(X, self.final_features)

        # Step 6: Scale
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
