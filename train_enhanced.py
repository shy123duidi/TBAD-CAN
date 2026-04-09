"""
Scenario B: Enhanced Method Training & Evaluation
Train: 1500 normal / 100 real malicious + synthetic malicious
Test: Identical test set as Scenario A
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
from models.transformer_model import TransformerAutoregressive
from utils.data_loader import load_can_data, create_sequences, split_data, normalize_data
from utils.feature_extractor import combine_all_features
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve, print_metrics_report, save_metrics


def load_synthetic_data(gan_model_path, data_shape, n_samples):
    """
    生成合成数据
    """
    print("Loading Transformer model for synthetic data generation...")
    gan = TransformerAutoregressive(
        seq_len=data_shape[1],
        feature_dim=data_shape[2],
        d_model=512,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        device='cuda'
    )
    gan.load_model(gan_model_path)
    
    print(f"Generating {n_samples} synthetic malware samples...")
    synthetic_data = gan.generate(n_samples, denormalize=True, temperature=0.5)
    print(f"Generated shape: {synthetic_data.shape}")
    
    return synthetic_data


def load_train_split_from_indices(features, labels, split_dir, window_size=5):
    """
    Load training split from saved indices
    """

    train_normal_idx = np.load(os.path.join(split_dir, 'train_normal_indices.npy'))
    train_malicious_idx = np.load(os.path.join(split_dir, 'train_malicious_indices.npy'))
    
    print(f"\nLoading training split from {split_dir}:")
    print(f"  Normal training indices: {len(train_normal_idx)}")
    print(f"  Malicious training indices: {len(train_malicious_idx)}")


    train_normal_features = features[train_normal_idx]
    train_normal_labels = labels[train_normal_idx]
    train_malicious_features = features[train_malicious_idx]
    train_malicious_labels = labels[train_malicious_idx]
    

    X_normal_train, y_normal_train = create_sequences(
        train_normal_features, train_normal_labels, 
        window_size=window_size, stride=1
    )
    
    X_real_malicious, y_real_malicious = create_sequences(
        train_malicious_features, train_malicious_labels,
        window_size=window_size, stride=1
    )
    
    return X_normal_train, y_normal_train, X_real_malicious, y_real_malicious


def main():
    parser = argparse.ArgumentParser(description='Enhanced CAN Anomaly Detection - Scenario B')
    parser.add_argument('--data_file', type=str, default='data/dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--gan_model_path', type=str, default='results/transformer_autoregressive_best.pth',
                        help='Path to trained Transformer model')
    parser.add_argument('--baseline_dir', type=str, default='results/baseline',
                        help='Directory containing Scenario A results (for test data)')
    parser.add_argument('--output_dir', type=str, default='results/enhanced',
                        help='Directory to save results')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size for sequence creation')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'svm', 'logistic', 'knn', 'mlp'],
                        help='Type of baseline model')
    parser.add_argument('--train_normal', type=int, default=1500,
                        help='Number of normal samples for training')
    parser.add_argument('--real_malicious', type=int, default=100,
                        help='Number of real malicious samples for training')
    parser.add_argument('--synthetic_malicious', type=int, default=1400,
                        help='Number of synthetic malicious samples to generate')
    parser.add_argument('--do_grid_search', action='store_true',
                        help='Perform grid search for hyperparameter tuning')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" " * 20 + "Scenario B: Enhanced Anomaly Detection")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Data file: {args.data_file}")
    print(f"  - GAN model: {args.gan_model_path}")
    print(f"  - Baseline dir: {args.baseline_dir}")
    print(f"  - Training: {args.train_normal} Normal + {args.real_malicious} Real Malicious + {args.synthetic_malicious} Synthetic Malicious")
    print(f"  - Testing:  Using test data from Scenario A")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Model: {args.model_type}")
    print("="*70)
    
    # 1. Load real CAN data
    print("\nStep 1: Loading real CAN data")
    features, labels = load_can_data(args.data_file, include_labels=True)
    
    # 2. Load test data from Scenario A
    print("\nStep 2: Loading test data from Scenario A")
    test_data_path = os.path.join(args.baseline_dir, 'X_test.npy')
    test_labels_path = os.path.join(args.baseline_dir, 'y_test.npy')
    
    if not os.path.exists(test_data_path) or not os.path.exists(test_labels_path):
        print(f"Error: Test data not found in {args.baseline_dir}")
        print("Please run Scenario A first to generate test data.")
        return
    
    X_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)
    
    print(f"  Loaded test data: {X_test.shape}")
    print(f"  Loaded test labels: {y_test.shape}")
    print(f"    Normal: {np.sum(y_test == 0)}")
    print(f"    Malicious: {np.sum(y_test == 1)}")
    
    # 3. Load training split from Scenario A
    print("\nStep 3: Loading training split from Scenario A")
    X_normal_train, y_normal_train, X_real_malicious, y_real_malicious = load_train_split_from_indices(
        features, labels, args.baseline_dir, args.window_size
    )
    
    print(f"\n  Real normal training sequences: {len(X_normal_train)}")
    print(f"  Real malicious training sequences: {len(X_real_malicious)}")
    
    # 4. Generate synthetic malicious data
    print("\nStep 4: Generating synthetic malicious data")

    seq_shape = (args.synthetic_malicious, args.window_size, X_normal_train.shape[-1])

    synthetic_sequences = load_synthetic_data(
        args.gan_model_path, seq_shape, args.synthetic_malicious
    )

    if synthetic_sequences.shape[1] != args.window_size:
        print(f"Warning: Synthetic sequence length mismatch. Adjusting...")
        if synthetic_sequences.shape[1] > args.window_size:
            synthetic_sequences = synthetic_sequences[:, :args.window_size, :]
        else:
            pad_width = args.window_size - synthetic_sequences.shape[1]
            synthetic_sequences = np.pad(synthetic_sequences, ((0, 0), (0, pad_width), (0, 0)), mode='edge')

    if len(synthetic_sequences) > args.synthetic_malicious:
        synthetic_sequences = synthetic_sequences[:args.synthetic_malicious]
    
    synthetic_labels = np.ones(len(synthetic_sequences))
    
    print(f"  Synthetic malicious sequences: {len(synthetic_sequences)}")
    
    # 5. Combine training data
    print("\nStep 5: Combining training data")

    X_malicious_train = np.vstack([X_real_malicious, synthetic_sequences])
    y_malicious_train = np.hstack([y_real_malicious, synthetic_labels])

    X_train_all = np.vstack([X_normal_train, X_malicious_train])
    y_train_all = np.hstack([y_normal_train, y_malicious_train])
    
    print(f"\n  Total training sequences: {len(X_train_all)}")
    print(f"    Normal: {np.sum(y_train_all == 0)}")
    print(f"    Malicious: {np.sum(y_train_all == 1)}")
    print(f"      - Real: {len(X_real_malicious)}")
    print(f"      - Synthetic: {len(synthetic_sequences)}")
    
    # 6. Extract advanced features
    print("\nStep 6: Extracting advanced features")
    X_train_features = combine_all_features(X_train_all)
    X_test_features = combine_all_features(X_test)
    
    # 7. Train model
    print(f"\nStep 7: Training {args.model_type} model with enhanced data")
    detector = BaselineDetector(model_type=args.model_type, random_state=args.random_seed)
    
    if args.do_grid_search:
        detector.grid_search(X_train_features, y_train_all)
    else:
        detector.train(X_train_features, y_train_all)
    
    # 8. Predict
    print("\nStep 8: Making predictions")
    y_pred = detector.predict(X_test_features)
    y_scores = detector.predict_proba(X_test_features)[:, 1]
    
    # 9. Evaluate
    print("\nStep 9: Evaluation")
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    print_metrics_report(metrics)
    
    # 10. Save results
    print("\nStep 10: Saving results")
    
    detector.save(os.path.join(args.output_dir, 'model.pkl'))
    
    metrics['model_type'] = args.model_type
    metrics['train_normal'] = args.train_normal
    metrics['real_malicious'] = args.real_malicious
    metrics['synthetic_malicious'] = args.synthetic_malicious
    metrics['window_size'] = args.window_size
    metrics['random_seed'] = args.random_seed
    save_metrics(metrics, os.path.join(args.output_dir, 'metrics.json'))
    
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_scores, save_path=os.path.join(args.output_dir, 'roc_curve.png'))
    
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(args.output_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(args.output_dir, 'y_scores.npy'), y_scores)
    np.save(os.path.join(args.output_dir, 'synthetic_samples.npy'), synthetic_sequences[:100])
    
    print(f"\n✅ Scenario B completed! Results saved to {args.output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
