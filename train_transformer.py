"""
Train Transformer-based autoregressive model for malicious data generation
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer_model import TransformerAutoregressive
from utils.data_loader import load_can_data, create_sequences
import warnings
warnings.filterwarnings('ignore')


def load_malware_data(file_path, window_size=5, stride=1):
    """Load malware data and create sequences for training"""
    print(f"\n{'='*60}")
    print(f"Loading Malware Data")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    

    features, labels = load_can_data(file_path, include_labels=True)
    

    malicious_indices = np.where(labels == 1)[0]
    malicious_features = features[malicious_indices]
    malicious_labels = labels[malicious_indices]
    
    print(f"  Total malicious samples: {len(malicious_features)}")
    
    sequences, seq_labels = create_sequences(
        malicious_features, 
        malicious_labels, 
        window_size=window_size, 
        stride=stride
    )
    
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  All sequences are malicious: {np.all(seq_labels == 1)}")
    
    return sequences


def save_training_history(loss_history, output_dir):
    """
    Save training history to CSV file
    """
    if loss_history and 'train' in loss_history:
        history_df = pd.DataFrame({
            'epoch': range(1, len(loss_history['train']) + 1),
            'train_loss': loss_history['train']
        })
        history_path = os.path.join(output_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"  Training history saved to {history_path}")
        return history_df
    else:
        print("  Warning: No training history to save")
        return None


def visualize_comparison(real_data, synthetic_data, output_dir, n_samples=1000):
    """
    Visualize comparison between real and synthetic data
    """
    print("\n" + "="*60)
    print("Visualizing Real vs Synthetic Data")
    print("="*60)

    n_samples = min(n_samples, len(real_data), len(synthetic_data))

    np.random.seed(42)
    real_indices = np.random.choice(len(real_data), n_samples, replace=False)
    synth_indices = np.random.choice(len(synthetic_data), n_samples, replace=False)
    
    real_sample = real_data[real_indices]
    synth_sample = synthetic_data[synth_indices]

    real_flat = real_sample.reshape(n_samples, -1)
    synth_flat = synth_sample.reshape(n_samples, -1)
    
    n_features = real_data.shape[2]
    
    # 1. Feature distribution comparison plot
    print("\n1. Plotting feature distributions...")
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features > 1 else [axes]
    
    for i in range(n_features):
        ax = axes[i]
        real_values = real_sample[:, :, i].flatten()
        synth_values = synth_sample[:, :, i].flatten()
        
        ax.hist(real_values, bins=50, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(synth_values, bins=50, alpha=0.5, label='Synthetic', density=True, color='red')
        ax.set_title(f'Feature {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: feature_distributions.png")
    
    # 2. PCA visualization plot
    print("\n2. Performing PCA visualization...")
    combined_data = np.vstack([real_flat, synth_flat])
    labels = np.array(['Real'] * len(real_flat) + ['Synthetic'] * len(synth_flat))
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(combined_data)
    
    plt.figure(figsize=(10, 8))
    for label, color in [('Real', 'blue'), ('Synthetic', 'red')]:
        mask = labels == label
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=color, label=label, alpha=0.5, s=10)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA Visualization: Real vs Synthetic Data')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pca_visualization.png")
    
    print("\n✅ Visualization completed!")


def calculate_kl_divergence(p, q, bins=50):
    """
    Calculate KL divergence between two probability distributions
    """
    hist_p, bins_p = np.histogram(p, bins=bins, density=True)
    hist_q, bins_q = np.histogram(q, bins=bins, density=True)
    
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    
    hist_p = hist_p / hist_p.sum()
    hist_q = hist_q / hist_q.sum()
    
    kl_div = np.sum(hist_p * np.log(hist_p / hist_q))
    
    return kl_div


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Autoregressive Model on malicious CAN data')
    parser.add_argument('--data_file', type=str, default='data/malicious_dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size for sequence creation')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sequence creation')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--generate_samples', type=int, default=2000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--generate_steps', type=int, default=None,
                        help='Number of steps to generate (default: window_size)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip visualization to save time')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" " * 15 + "Transformer Autoregressive Model Training (Improved)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Data file: {args.data_file}")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Model dimension: {args.d_model}")
    print(f"  - Attention heads: {args.nhead}")
    print(f"  - Transformer layers: {args.num_layers}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Weight Decay: {args.weight_decay}")
    print(f"  - Device: {args.device}")
    print("="*70)
    

    print("\nStep 1: Loading Malware Data")
    malware_data = load_malware_data(
        args.data_file, args.window_size, args.stride
    )

    if len(malware_data) == 0:
        print("Error: No malicious data found for training!")
        return
    
    print(f"\n  Training data shape: {malware_data.shape}")
    print(f"  Training data range: [{malware_data.min():.2f}, {malware_data.max():.2f}]")

    print("\nStep 2: Initializing Transformer Autoregressive Model")
    model = TransformerAutoregressive(
        seq_len=args.window_size,
        feature_dim=malware_data.shape[-1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device
    )

    print("\nStep 3: Training Model (This may take a while...)")
    start_time = time.time()
    
    model.fit(
        real_data=malware_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        verbose=True,
        save_scaler=True,
        save_dir=args.output_dir,
        save_freq=50
    )
    
    print(f"\nTraining time: {time.time() - start_time:.2f} seconds")

    print("\nStep 4: Generating Synthetic Malware")
    synthetic_malware = model.generate(
        args.generate_samples, 
        n_steps=args.generate_steps,
        denormalize=True
    )
    print(f"  Generated {len(synthetic_malware)} samples")
    print(f"  Shape: {synthetic_malware.shape}")
    print(f"  Range: [{synthetic_malware.min():.2f}, {synthetic_malware.max():.2f}]")

    print("\nStep 5: Saving Results")

    synthetic_path = os.path.join(args.output_dir, 'synthetic_malware_autoregressive.npy')
    np.save(synthetic_path, synthetic_malware)
    print(f"  Synthetic data saved to {synthetic_path}")

    save_synthetic_as_csv(synthetic_malware, args.output_dir)

    model_path = os.path.join(args.output_dir, 'transformer.pth')
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")

    loss_plot_path = os.path.join(args.output_dir, 'transformer_autoregressive_loss.png')
    model.plot_losses(save_path=loss_plot_path)
    print(f"  Loss plot saved to {loss_plot_path}")

    save_training_history(model.loss_history, args.output_dir)

    if not args.skip_visualization:
        print("\n" + "="*70)
        print("Step 6: Visualizing Real vs Synthetic Data")
        print("="*70)
        visualize_comparison(malware_data, synthetic_malware, args.output_dir)

    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"  Original malicious samples: {len(malware_data)}")
    print(f"  Generated synthetic samples: {len(synthetic_malware)}")
    
    if model.loss_history and 'train' in model.loss_history:
        print(f"  Final Training Loss: {model.loss_history['train'][-1]:.4f}")
        print(f"  Best Training Loss: {min(model.loss_history['train']):.4f}")
    
    print(f"\n  Synthetic Data Statistics:")
    print(f"    Mean: {synthetic_malware.mean():.2f}")
    print(f"    Std: {synthetic_malware.std():.2f}")
    print(f"    Min: {synthetic_malware.min():.2f}")
    print(f"    Max: {synthetic_malware.max():.2f}")
    print("="*70)
    
    print("\n✅ Transformer Autoregressive Model training completed!")
    print(f"Results saved to {args.output_dir}")
    
    return synthetic_malware


def save_synthetic_as_csv(synthetic_data, output_dir):
    """
    Save synthetic data as CSV file
    """
    n_samples, seq_len, n_features = synthetic_data.shape
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    columns = []
    for t in range(seq_len):
        for f in feature_names:
            columns.append(f't{t}_{f}')
    
    synthetic_flat = synthetic_data.reshape(n_samples, -1)
    
    df_synthetic = pd.DataFrame(synthetic_flat, columns=columns)
    
    df_synthetic['label'] = 'T'
    df_synthetic['label_numeric'] = 1
    df_synthetic['sample_id'] = range(1, n_samples + 1)
    df_synthetic['sample_type'] = 'synthetic_malicious_autoregressive'
    
    df_synthetic['mean_value'] = synthetic_data.mean(axis=(1, 2))
    df_synthetic['std_value'] = synthetic_data.std(axis=(1, 2))
    
    csv_path = os.path.join(output_dir, 'synthetic_malware_autoregressive.csv')
    df_synthetic.to_csv(csv_path, index=False)
    print(f"  Synthetic data (CSV) saved to {csv_path}")
    
    save_detailed_format(synthetic_data, output_dir)


def save_detailed_format(synthetic_data, output_dir):
    """
    Save detailed format of synthetic data as CSV file
    """
    n_samples, seq_len, n_features = synthetic_data.shape
    
    detailed_data = []
    for sample_idx in range(min(n_samples, 100)):
        for time_step in range(seq_len):
            row = {
                'sample_id': sample_idx + 1,
                'time_step': time_step,
                'label': 'T',
                'label_numeric': 1,
                'sample_type': 'synthetic_malicious_autoregressive'
            }
            for feat_idx in range(n_features):
                row[f'feature_{feat_idx}'] = synthetic_data[sample_idx, time_step, feat_idx]
            detailed_data.append(row)
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        detailed_path = os.path.join(output_dir, 'synthetic_malware_autoregressive_detailed.csv')
        df_detailed.to_csv(detailed_path, index=False)
        print(f"  Detailed format saved to {detailed_path}")


if __name__ == "__main__":
    main()
