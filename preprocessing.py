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
    This function handles IP encoding and protocol encoding using LabelEncoder.

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

    categorical_cols = ['src_ip', 'dst_ip', 'protocol']

    for col in categorical_cols:
        if col in df.columns:
            if fit_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            elif col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))

    # Handle infinite values
    df = df.replace([np.inf, -np.inf], 0)

    # Fill missing values
    df = df.fillna(0)

    return df, label_encoders


def remove_leaky_features(X):
    """
    Remove features that are too perfect indicators (cause data leakage).
    Based on notebook analysis, is_port_6200 and is_ftp_data_port are leaky.

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
    This class encapsulates all preprocessing steps in the correct order.
    Matches the exact preprocessing steps from the notebook.
    """

    def __init__(self):
        self.label_encoders = None
        self.scaler = None
        self.final_features = None
        self.removed_features = {
            'leaky': [],
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

        # Step 1: Encode categorical features (src_ip, dst_ip, protocol) using LabelEncoder
        X, self.label_encoders = preprocess_raw_data(X, fit_encoders=True)

        # Step 2: Remove leaky features (is_port_6200, is_ftp_data_port)
        X, leaky = remove_leaky_features(X)
        self.removed_features['leaky'] = leaky

        # Step 3: Remove zero variance features
        X, zero_var = remove_zero_variance_features(X)
        self.removed_features['zero_variance'] = zero_var

        # Step 4: Remove weak features (correlation < 0.05)
        X, weak = remove_weak_features(X, y, correlation_threshold=0.05)
        self.removed_features['weak_correlation'] = weak

        # Step 5: Remove highly correlated features (correlation > 0.95)
        X, high_corr = remove_highly_correlated_features(X, y, correlation_threshold=0.95)
        self.removed_features['high_correlation'] = high_corr

        # Store final features
        self.final_features = X.columns.tolist()

        # Step 6: Fit scaler (will be applied after train-test split)
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

        # Step 1: Encode categorical features
        X, _ = preprocess_raw_data(df, label_encoders=self.label_encoders, fit_encoders=False)

        # Step 2-5: Remove features (using saved list from fit)
        features_to_remove = (
            self.removed_features['leaky'] +
            self.removed_features['zero_variance'] +
            self.removed_features['weak_correlation'] +
            self.removed_features['high_correlation']
        )

        for feat in features_to_remove:
            if feat in X.columns:
                X = X.drop(columns=[feat])

        # Ensure correct feature order
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
