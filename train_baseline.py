"""
Scenario A: Baseline Training & Evaluation
Train: 1500 normal / 100 malicious
Test: 500 normal / 500 malicious
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.baseline_models import BaselineDetector
from utils.data_loader import load_can_data, create_sequences, split_data, normalize_data
from utils.feature_extractor import combine_all_features
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve, print_metrics_report, save_metrics


def create_fixed_split(features, labels, window_size=5, 
                       train_normal=1500, train_malicious=100,
                       test_normal=500, test_malicious=500,
                       random_state=42, save_split=True, output_dir=None):
    """
    Create fixed split for training and testing, and save split information
    """
    np.random.seed(random_state)
    

    normal_indices = np.where(labels == 0)[0]
    malicious_indices = np.where(labels == 1)[0]
    
    print(f"\nAvailable data:")
    print(f"  Total normal samples: {len(normal_indices)}")
    print(f"  Total malicious samples: {len(malicious_indices)}")
    

    if len(normal_indices) < train_normal + test_normal:
        print(f"Warning: Only {len(normal_indices)} normal samples available. Adjusting...")
        test_normal = max(0, len(normal_indices) - train_normal)
        train_normal = len(normal_indices) - test_normal
    
    if len(malicious_indices) < train_malicious + test_malicious:
        print(f"Warning: Only {len(malicious_indices)} malicious samples available. Adjusting...")
        test_malicious = max(0, len(malicious_indices) - train_malicious)
        train_malicious = len(malicious_indices) - test_malicious
    

    train_normal_idx = np.random.choice(normal_indices, train_normal, replace=False)
    remaining_normal = np.setdiff1d(normal_indices, train_normal_idx)
    test_normal_idx = np.random.choice(remaining_normal, min(test_normal, len(remaining_normal)), replace=False)
    
    train_malicious_idx = np.random.choice(malicious_indices, train_malicious, replace=False)
    remaining_malicious = np.setdiff1d(malicious_indices, train_malicious_idx)
    test_malicious_idx = np.random.choice(remaining_malicious, min(test_malicious, len(remaining_malicious)), replace=False)
    

    if save_split and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'train_normal_indices.npy'), train_normal_idx)
        np.save(os.path.join(output_dir, 'train_malicious_indices.npy'), train_malicious_idx)
        np.save(os.path.join(output_dir, 'test_normal_indices.npy'), test_normal_idx)
        np.save(os.path.join(output_dir, 'test_malicious_indices.npy'), test_malicious_idx)
        print(f"  Saved split indices to {output_dir}")
    

    train_indices = np.concatenate([train_normal_idx, train_malicious_idx])
    test_indices = np.concatenate([test_normal_idx, test_malicious_idx])
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    print(f"\nSelected data:")
    print(f"  Training - Normal: {len(train_normal_idx)}, Malicious: {len(train_malicious_idx)}")
    print(f"  Testing  - Normal: {len(test_normal_idx)}, Malicious: {len(test_malicious_idx)}")
    

    X_train, y_train = create_sequences(train_features, train_labels, window_size=window_size, stride=1)
    X_test, y_test = create_sequences(test_features, test_labels, window_size=window_size, stride=1)
    
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description='Baseline CAN Anomaly Detection - Scenario A')
    parser.add_argument('--data_file', type=str, default='data/dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='results/baseline',
                        help='Directory to save results')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size for sequence creation')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'svm', 'logistic', 'knn', 'mlp'],
                        help='Type of baseline model')
    parser.add_argument('--train_normal', type=int, default=1500,
                        help='Number of normal samples for training')
    parser.add_argument('--train_malicious', type=int, default=100,
                        help='Number of real malicious samples for training')
    parser.add_argument('--test_normal', type=int, default=500,
                        help='Number of normal samples for testing')
    parser.add_argument('--test_malicious', type=int, default=500,
                        help='Number of malicious samples for testing')
    parser.add_argument('--do_grid_search', action='store_true',
                        help='Perform grid search for hyperparameter tuning')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" " * 20 + "Scenario A: Baseline Anomaly Detection")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Data file: {args.data_file}")
    print(f"  - Training: {args.train_normal} Normal + {args.train_malicious} Malicious")
    print(f"  - Testing:  {args.test_normal} Normal + {args.test_malicious} Malicious")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Model: {args.model_type}")
    print("="*70)
    
    # 1. Load CAN data
    print("\nStep 1: Loading CAN data")
    features, labels = load_can_data(args.data_file, include_labels=True)
    
    # 2. Create fixed train/test split
    print("\nStep 2: Creating fixed train/test split")
    X_train, X_test, y_train, y_test = create_fixed_split(
        features, labels, 
        window_size=args.window_size,
        train_normal=args.train_normal,
        train_malicious=args.train_malicious,
        test_normal=args.test_normal,
        test_malicious=args.test_malicious,
        random_state=args.random_seed,
        save_split=True,
        output_dir=args.output_dir
    )
    
    print(f"\nFinal datasets:")
    print(f"  Training sequences: {len(X_train)} (Normal: {np.sum(y_train == 0)}, Malicious: {np.sum(y_train == 1)})")
    print(f"  Testing sequences:  {len(X_test)} (Normal: {np.sum(y_test == 0)}, Malicious: {np.sum(y_test == 1)})")
    
    # Save test data for Scenario B use
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    print(f"  Saved test data for Scenario B")
    
    # 3. Extract advanced features
    print("\nStep 3: Extracting advanced features")
    X_train_features = combine_all_features(X_train)
    X_test_features = combine_all_features(X_test)
    
    # 4. Train model
    print(f"\nStep 4: Training {args.model_type} model")
    detector = BaselineDetector(model_type=args.model_type, random_state=args.random_seed)
    
    if args.do_grid_search:
        detector.grid_search(X_train_features, y_train)
    else:
        detector.train(X_train_features, y_train)
    
    # 5. Predict
    print("\nStep 5: Making predictions")
    y_pred = detector.predict(X_test_features)
    y_scores = detector.predict_proba(X_test_features)[:, 1]
    
    # 6. Evaluation
    print("\nStep 6: Evaluation")
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    print_metrics_report(metrics)
    
    # 7. Save results
    print("\nStep 7: Saving results")
    
    detector.save(os.path.join(args.output_dir, 'model.pkl'))
    
    metrics['model_type'] = args.model_type
    metrics['train_normal'] = args.train_normal
    metrics['train_malicious'] = args.train_malicious
    metrics['test_normal'] = args.test_normal
    metrics['test_malicious'] = args.test_malicious
    metrics['window_size'] = args.window_size
    metrics['random_seed'] = args.random_seed
    save_metrics(metrics, os.path.join(args.output_dir, 'metrics.json'))
    
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_scores, save_path=os.path.join(args.output_dir, 'roc_curve.png'))

    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(args.output_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(args.output_dir, 'y_scores.npy'), y_scores)
    
    print(f"\n✅ Scenario A completed! Results saved to {args.output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
