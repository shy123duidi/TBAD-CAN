"""
Metrics module for evaluating model performance
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json


def calculate_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate classification metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['roc_auc'] = 0.0
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Plot ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.show()
    plt.close()


def print_metrics_report(metrics):
    """
    Print classification metrics report
    """
    print("\n" + "="*50)
    print("Classification Metrics")
    print("="*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
    print("="*50)


def save_metrics(metrics, save_path):
    """
    Save classification metrics to JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_path}")


def plot_training_curves(history, save_path=None):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # loss curve
    axes[0].plot(history.get('epochs', range(len(history.get('loss', [])))), 
                 history.get('loss', []), label='Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if 'val_loss' in history:
        axes[1].plot(history.get('epochs', range(len(history.get('val_loss', [])))), 
                     history.get('val_loss', []), label='Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    plt.show()
    plt.close()
