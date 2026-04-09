"""
CAN bus data loader module
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def extract_frame_features(can_id, data_bytes):
    """
    Extract features from a single CAN frame
    """
    if isinstance(can_id, str):
        try:
            can_id_val = int(can_id, 16)
        except:
            can_id_val = 0
    else:
        try:
            can_id_val = float(can_id)
        except:
            can_id_val = 0
    

    bytes_vals = []
    for b in data_bytes:
        if isinstance(b, str):
            try:
                bytes_vals.append(int(b, 16))
            except:
                bytes_vals.append(0)
        else:
            try:
                bytes_vals.append(float(b))
            except:
                bytes_vals.append(0)
    

    data_sum = sum(bytes_vals)
    data_mean = np.mean(bytes_vals)
    data_std = np.std(bytes_vals)
    data_max = max(bytes_vals)
    data_min = min(bytes_vals)
    

    if len(bytes_vals) > 1:
        diff_vals = np.diff(bytes_vals)
        diff_mean = np.mean(np.abs(diff_vals))
        diff_std = np.std(diff_vals)
    else:
        diff_mean = 0
        diff_std = 0
    

    features = [
        float(can_id_val), 
        float(data_sum), 
        float(data_mean), 
        float(data_std), 
        float(data_max), 
        float(data_min),
        float(diff_mean), 
        float(diff_std), 
        float(len(bytes_vals))
    ]

    features = [0.0 if np.isnan(f) else f for f in features]
    features = [0.0 if np.isinf(f) else f for f in features]
    
    return features


def load_can_data(file_path, include_labels=True):
    """
    Load CAN bus data from CSV file
    """
    print(f"\n{'='*60}")
    print(f"Loading CAN data from: {file_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path, header=None)
    print(f"Raw data shape: {df.shape}")
    

    timestamps = df.iloc[:, 0].values
    can_ids = df.iloc[:, 1].values
    dlc = df.iloc[:, 2].values
    data_bytes = df.iloc[:, 3:11].values
    

    if include_labels and len(df.columns) >= 12:
        labels_raw = df.iloc[:, 11].values
        labels = []
        for label in labels_raw:
            if isinstance(label, str):
                if label.upper() == 'R':
                    labels.append(0)  # normal
                elif label.upper() == 'T':
                    labels.append(1)  # malicious
                else:
                    labels.append(0)
            else:
                labels.append(0)
        labels = np.array(labels, dtype=np.int32)
        print(f"Labels distribution - Normal (R): {np.sum(labels == 0)}")
        print(f"               - Malicious (T): {np.sum(labels == 1)}")
    else:
        labels = None
    

    features = []
    for i in range(len(df)):
        row_features = extract_frame_features(can_ids[i], data_bytes[i])
        features.append(row_features)
    
    features = np.array(features, dtype=np.float32)
    

    if np.isnan(features).any():
        print("Warning: Features contain NaN. Filling with 0.")
        features = np.nan_to_num(features, nan=0.0)
    
    if np.isinf(features).any():
        print("Warning: Features contain inf. Filling with 0.")
        features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
    
    return features, labels


def create_sequences(features, labels, window_size=5, stride=1):
    """
    Create time series sequences from features
    """
    n_samples = len(features)
    sequences = []
    sequence_labels = []


    features = np.nan_to_num(features, nan=0.0)
    
    for i in range(0, n_samples - window_size + 1, stride):
        seq = features[i:i + window_size]
        seq_label = 1 if np.sum(labels[i:i + window_size]) > 0 else 0
        sequences.append(seq)
        sequence_labels.append(seq_label)
    
    sequences = np.array(sequences, dtype=np.float32)
    sequence_labels = np.array(sequence_labels, dtype=np.int32)
    
    print(f"Created {len(sequences)} sequences (window_size={window_size})")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Sequence labels - Normal: {np.sum(sequence_labels == 0)}, Anomaly: {np.sum(sequence_labels == 1)}")
    
    return sequences, sequence_labels


def split_data(sequences, labels, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets with stratified sampling
    """
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples (Normal: {np.sum(y_train == 0)}, Anomaly: {np.sum(y_train == 1)})")
    print(f"  Test:  {X_test.shape[0]} samples (Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)})")
    
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test, scaler=None):
    """
    Normalize data features
    """
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    if scaler is None:
        scaler = StandardScaler()
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(original_shape)
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_flat = scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
    else:
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_flat = scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
    
    return X_train, X_test, scaler
