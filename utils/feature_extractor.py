"""
Feature extractor module
"""

import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def extract_advanced_features(sequences):
    """
    Extract advanced features from sequences of CAN frames
    """
    n_samples, window_size, n_features = sequences.shape
    advanced_features = []
    
    for seq in sequences:
        seq_features = []
        

        for dim in range(n_features):
            dim_values = seq[:, dim]
            dim_values = np.nan_to_num(dim_values, nan=0.0)
            seq_features.extend([
                np.mean(dim_values),
                np.std(dim_values),
                np.max(dim_values),
                np.min(dim_values),
                np.percentile(dim_values, 25),
                np.percentile(dim_values, 75)
            ])
        

        for dim in range(n_features):
            dim_values = seq[:, dim]
            dim_values = np.nan_to_num(dim_values, nan=0.0)
            x = np.arange(len(dim_values))
            if len(dim_values) > 1:
                slope = np.polyfit(x, dim_values, 1)[0]
                seq_features.append(slope)
            else:
                seq_features.append(0)
        

        if n_features > 0:
            can_ids = seq[:, 0]
            can_ids = np.nan_to_num(can_ids, nan=0.0)
            unique, counts = np.unique(can_ids, return_counts=True)
            probs = counts / len(can_ids)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            seq_features.append(entropy)
        

        if n_features > 1:
            seq_clean = np.nan_to_num(seq, nan=0.0)
            corr_matrix = np.corrcoef(seq_clean.T)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            seq_features.append(np.mean(upper_tri))
            seq_features.append(np.std(upper_tri))
        
        advanced_features.append(seq_features)
    
    return np.array(advanced_features)


def extract_temporal_features(sequences):
    """
    Extract temporal features from sequences of CAN frames
    """
    n_samples, window_size, n_features = sequences.shape
    temporal_features = []
    
    for seq in sequences:
        seq_features = []
        
        for dim in range(n_features):
            dim_values = seq[:, dim]
            dim_values = np.nan_to_num(dim_values, nan=0.0)
            

            if len(dim_values) > 1:
                if np.std(dim_values) > 1e-6 and np.std(dim_values[:-1]) > 1e-6:
                    autocorr = np.corrcoef(dim_values[:-1], dim_values[1:])[0, 1]
                else:
                    autocorr = 0
            else:
                autocorr = 0
            seq_features.append(autocorr)
            

            if len(dim_values) > 1:
                rate_of_change = np.mean(np.abs(np.diff(dim_values)))
            else:
                rate_of_change = 0
            seq_features.append(rate_of_change)
        
        temporal_features.append(seq_features)
    
    return np.array(temporal_features)


def extract_frequency_features(sequences):
    """
    Extract frequency features from sequences of CAN frames
    """
    n_samples, window_size, n_features = sequences.shape
    freq_features = []
    
    for seq in sequences:
        seq_features = []
        
        for dim in range(n_features):
            dim_values = seq[:, dim]
            dim_values = np.nan_to_num(dim_values, nan=0.0)
            
            # FFT
            fft_vals = np.fft.fft(dim_values)
            fft_magnitude = np.abs(fft_vals)
            

            if len(fft_magnitude) > 1:
                if len(fft_magnitude) > 5:
                    magnitude_subset = fft_magnitude[1:5]
                else:
                    magnitude_subset = fft_magnitude[1:]
                seq_features.append(np.mean(magnitude_subset))
                seq_features.append(np.std(magnitude_subset))
                seq_features.append(np.max(magnitude_subset))
            else:
                seq_features.extend([0, 0, 0])
        
        freq_features.append(seq_features)
    
    return np.array(freq_features)


def combine_all_features(sequences):
    """
    Combine all features from sequences of CAN frames, ensuring no NaN values
    """
    n_samples, window_size, n_features = sequences.shape
    

    seq_flat = sequences.reshape(n_samples, -1)
    seq_flat = np.nan_to_num(seq_flat, nan=0.0)
    

    adv_features = extract_advanced_features(sequences)
    temp_features = extract_temporal_features(sequences)
    freq_features = extract_frequency_features(sequences)

    adv_features = np.nan_to_num(adv_features, nan=0.0)
    temp_features = np.nan_to_num(temp_features, nan=0.0)
    freq_features = np.nan_to_num(freq_features, nan=0.0)
    

    combined = np.hstack([seq_flat, adv_features, temp_features, freq_features])

    if np.isnan(combined).any():
        print("Warning: Still found NaN in combined features. Filling with 0.")
        combined = np.nan_to_num(combined, nan=0.0)
    if np.isinf(combined).any():
        print("Warning: Found inf values in combined features. Replacing with 0.")
        combined = np.nan_to_num(combined, posinf=0.0, neginf=0.0)
    
    print(f"Combined features shape: {combined.shape}")
    print(f"  - Original flattened: {seq_flat.shape[1]}")
    print(f"  - Advanced features: {adv_features.shape[1]}")
    print(f"  - Temporal features: {temp_features.shape[1]}")
    print(f"  - Frequency features: {freq_features.shape[1]}")
    print(f"  - Data range: [{combined.min():.4f}, {combined.max():.4f}]")
    
    return combined
