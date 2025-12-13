"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing import (
    ip_to_int,
    preprocess_raw_data,
    remove_zero_variance_features,
    remove_weak_features,
    remove_highly_correlated_features,
    get_final_features,
    prepare_features_for_prediction,
    NetworkTrafficPreprocessor
)


class TestIPConversion:
    """Test IP address conversion functions."""

    def test_ip_to_int_valid(self):
        """Test conversion of valid IP addresses."""
        assert ip_to_int("192.168.1.1") == 3232235777
        assert ip_to_int("10.0.0.1") == 167772161
        assert ip_to_int("127.0.0.1") == 2130706433

    def test_ip_to_int_invalid(self):
        """Test conversion of invalid IP addresses."""
        assert ip_to_int("invalid") == 0
        assert ip_to_int("999.999.999.999") == 0
        assert ip_to_int("") == 0


class TestPreprocessRawData:
    """Test raw data preprocessing."""

    def test_preprocess_with_ips(self):
        """Test preprocessing with IP addresses."""
        df = pd.DataFrame({
            'src_ip': ['192.168.1.1', '10.0.0.1'],
            'dst_ip': ['192.168.1.2', '10.0.0.2'],
            'protocol': ['TCP', 'UDP']
        })

        result, encoder = preprocess_raw_data(df, fit_encoder=True)

        # Check IPs are converted to integers
        assert result['src_ip'].dtype == np.int64
        assert result['dst_ip'].dtype == np.int64
        assert result['src_ip'][0] == 3232235777

        # Check protocol is encoded
        assert result['protocol'].dtype == np.int64
        assert encoder is not None

    def test_preprocess_with_existing_encoder(self):
        """Test preprocessing with pre-fitted encoder."""
        df = pd.DataFrame({
            'protocol': ['TCP', 'UDP', 'TCP']
        })

        # Fit encoder
        encoder = LabelEncoder()
        encoder.fit(['TCP', 'UDP', 'ICMP'])

        # Use fitted encoder
        result, _ = preprocess_raw_data(df, label_encoder=encoder, fit_encoder=False)

        assert result['protocol'].dtype == np.int64
        assert len(result) == 3


class TestFeatureRemoval:
    """Test feature removal functions."""

    def test_remove_zero_variance(self):
        """Test removal of zero variance features."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 5, 5, 5, 5],  # Zero variance
            'feature3': [1, 1, 1, 1, 1],  # Zero variance
            'feature4': [1, 2, 1, 2, 1]
        })

        X_filtered, removed = remove_zero_variance_features(X)

        assert 'feature1' in X_filtered.columns
        assert 'feature4' in X_filtered.columns
        assert 'feature2' not in X_filtered.columns
        assert 'feature3' not in X_filtered.columns
        assert set(removed) == {'feature2', 'feature3'}

    def test_remove_weak_features(self):
        """Test removal of weakly correlated features."""
        X = pd.DataFrame({
            'strong': [1, 2, 3, 4, 5],
            'weak': [1, 1, 1, 1, 2],
            'medium': [1, 2, 2, 3, 4]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        X_filtered, removed = remove_weak_features(X, y, correlation_threshold=0.05)

        # Strong feature should remain
        assert 'strong' in X_filtered.columns

    def test_remove_highly_correlated(self):
        """Test removal of highly correlated features."""
        # Create highly correlated features
        X = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with f1
            'f3': [5, 4, 3, 2, 1]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        X_filtered, removed = remove_highly_correlated_features(X, y, correlation_threshold=0.95)

        # Should remove one of f1/f2
        assert len(X_filtered.columns) < len(X.columns)


class TestFinalFeatures:
    """Test final feature functions."""

    def test_get_final_features(self):
        """Test getting list of final features."""
        features = get_final_features()

        assert isinstance(features, list)
        assert len(features) == 14
        assert 'src_ip' in features
        assert 'is_port_22' in features
        assert 'label' not in features

    def test_prepare_features_correct_order(self):
        """Test preparing features in correct order."""
        final_features = get_final_features()

        # Create DataFrame with features in random order
        df = pd.DataFrame({
            feature: [0] for feature in reversed(final_features)
        })

        result = prepare_features_for_prediction(df, final_features)

        # Check order is correct
        assert list(result.columns) == final_features

    def test_prepare_features_missing(self):
        """Test error when features are missing."""
        df = pd.DataFrame({
            'src_ip': [1],
            'src_port': [80]
        })

        with pytest.raises(ValueError, match="Missing features"):
            prepare_features_for_prediction(df, get_final_features())


class TestNetworkTrafficPreprocessor:
    """Test complete preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        return pd.DataFrame({
            'src_ip': ['192.168.1.1', '10.0.0.1', '192.168.1.2'],
            'dst_ip': ['192.168.1.2', '10.0.0.2', '192.168.1.3'],
            'src_port': [12345, 54321, 8080],
            'dst_port': [22, 21, 80],
            'protocol': ['TCP', 'TCP', 'TCP'],
            'duration': [1.5, 2.3, 0.8],
            'total_packets': [100, 150, 80],
            'total_bytes': [5000, 7500, 4000],
            'min_packet_size': [64, 64, 64],
            'max_packet_size': [1500, 1500, 1500],
            'avg_packet_size': [50, 50, 50],
            'std_packet_size': [10, 10, 10],
            'syn_count': [1, 1, 1],
            'ack_count': [50, 75, 40],
            'fin_count': [1, 1, 1],
            'rst_count': [0, 0, 0],
            'psh_count': [20, 30, 15],
            'urg_count': [0, 0, 0],
            'packets_per_second': [66.67, 65.22, 100.0],
            'bytes_per_second': [3333.33, 3260.87, 5000.0],
            'bytes_per_packet': [50, 50, 50],
            'forward_packets': [50, 75, 40],
            'backward_packets': [50, 75, 40],
            'forward_bytes': [2500, 3750, 2000],
            'backward_bytes': [2500, 3750, 2000],
            'forward_backward_ratio': [1.0, 1.0, 1.0],
            'avg_iat': [0.01, 0.015, 0.01],
            'std_iat': [0.005, 0.007, 0.005],
            'min_iat': [0.001, 0.001, 0.001],
            'max_iat': [0.05, 0.06, 0.04],
            'syn_ack_ratio': [0.02, 0.013, 0.025],
            'is_port_22': [1, 0, 0],
            'is_port_6200': [0, 0, 0],
            'is_ftp_port': [0, 1, 0],
            'is_ftp_data_port': [0, 0, 0],
            'label': [0, 1, 0]
        })

    def test_fit(self, sample_data):
        """Test fitting the preprocessor."""
        preprocessor = NetworkTrafficPreprocessor()

        X, y = preprocessor.fit(sample_data, target_column='label')

        # Check that preprocessor components are fitted
        assert preprocessor.label_encoder is not None
        assert preprocessor.scaler is not None
        assert len(preprocessor.final_features) == 14

        # Check output shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)

    def test_transform(self, sample_data):
        """Test transforming new data."""
        preprocessor = NetworkTrafficPreprocessor()

        # Fit on data
        preprocessor.fit(sample_data, target_column='label')

        # Transform new data (without label)
        new_data = sample_data.drop('label', axis=1).iloc[:1]
        X_transformed = preprocessor.transform(new_data)

        # Check output is scaled numpy array
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[1] == 14

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = NetworkTrafficPreprocessor()

        X_scaled, y = preprocessor.fit_transform(sample_data, target_column='label')

        # Check outputs
        assert isinstance(X_scaled, np.ndarray)
        assert isinstance(y, pd.Series)
        assert X_scaled.shape[0] == len(y)
        assert X_scaled.shape[1] == 14

    def test_missing_target_column(self, sample_data):
        """Test error when target column is missing."""
        preprocessor = NetworkTrafficPreprocessor()

        with pytest.raises(ValueError, match="Target column .* not found"):
            preprocessor.fit(sample_data, target_column='nonexistent')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
