from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Enhanced visualizer for comprehensive model analysis and interpretation."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initialize the Visualizer with modern styling.

        Args:
            model: PyTorch model to visualize.
            device: Device to run computations on.
        """
        self.model: nn.Module = model
        self.device: torch.device = device
        
        # Set modern matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    def _get_color_palette(self, n_colors: int) -> NDArray:
        """Generate a modern color palette.

        Args:
            n_colors: Number of colors needed.

        Returns:
            Array of RGB colors.
        """
        if n_colors <= 10:
            # Use tab10 for small number of classes
            return plt.cm.tab10(np.linspace(0, 1, n_colors))
        else:
            # Use hsv for many classes
            return plt.cm.hsv(np.linspace(0, 0.9, n_colors))

    def plot_pca(self, dataloader: DataLoader, num_classes: int, 
                 save_path: str = 'pca_visualization.png', max_samples: int = 5000) -> None:
        """Generate enhanced PCA visualization with modern styling.

        Args:
            dataloader: DataLoader for the dataset.
            num_classes: Number of classes in the dataset.
            save_path: Path to save the visualization.
            max_samples: Maximum number of samples to use for PCA (default: 5000).
                        Prevents memory issues with large datasets.
        """
        print("\nGenerating PCA visualization...")

        self.model.eval()

        features_list: List[NDArray[np.float32]] = []
        labels_list: List[NDArray[np.int_]] = []
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                _, _, feats = self.model(X_batch)
                features_list.append(feats.cpu().numpy())
                labels_list.append(y_batch.numpy())
                
                total_samples += len(X_batch)
                if total_samples >= max_samples:
                    print(f"  [Info] Limiting to {max_samples} samples for PCA visualization")
                    break

        features = np.vstack(features_list)[:max_samples]
        labels = np.concatenate(labels_list)[:max_samples]
        prototypes = self.model.prototype.prototypes.detach().cpu().numpy()

        combined = np.vstack([features, prototypes])
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(combined)

        features_2d = embedded[:len(features)]
        prototypes_2d = embedded[len(features):]

        fig, ax = plt.subplots(figsize=(12, 9))
        colors = self._get_color_palette(num_classes)

        # Plot features with enhanced styling
        for i in range(num_classes):
            mask = labels == i
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[i]], alpha=0.6, s=50, label=f'Class {i}',
                      edgecolors='white', linewidth=0.5)

        # Plot prototypes with star markers
        protos_per_class = len(prototypes) // num_classes
        for i in range(num_classes):
            start = i * protos_per_class
            end = start + protos_per_class
            ax.scatter(prototypes_2d[start:end, 0], prototypes_2d[start:end, 1],
                      c=[colors[i]], marker='*', s=800, edgecolors='black', 
                      linewidth=2.5, zorder=10, label=f'Proto {i}')

        var1, var2 = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({var1:.1%} variance)', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Principal Component 2 ({var2:.1%} variance)', 
                     fontsize=13, fontweight='bold')
        ax.set_title('PCA Projection: Feature Space & Learned Prototypes', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Enhanced legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                 fontsize=10, framealpha=0.95, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add total variance explained
        total_var = var1 + var2
        ax.text(0.02, 0.98, f'Total Variance: {total_var:.1%}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_training_curves(self, history: Dict, save_path: str = 'training_curves.png') -> None:
        """Plot enhanced training history with multiple metrics.

        Args:
            history: Dictionary containing training history.
            save_path: Path to save the visualization.
        """
        print("\nGenerating training curves...")
        
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(epochs, history['loss'], linewidth=2.5, color='#e74c3c', 
                marker='o', markersize=4, markevery=max(1, len(epochs)//20))
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        
        # Add min loss annotation
        min_loss_idx = np.argmin(history['loss'])
        min_loss = history['loss'][min_loss_idx]
        ax1.annotate(f'Min: {min_loss:.4f}', 
                    xy=(min_loss_idx + 1, min_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Accuracy plot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(epochs, history['train_acc'], label='Train', linewidth=2.5, 
                color='#3498db', marker='s', markersize=4, 
                markevery=max(1, len(epochs)//20))
        ax2.plot(epochs, history['test_acc'], label='Test', linewidth=2.5, 
                color='#2ecc71', marker='^', markersize=4, 
                markevery=max(1, len(epochs)//20))
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, framealpha=0.95, edgecolor='gray')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor('#f8f9fa')
        
        # Add best accuracy annotation
        best_test_idx = np.argmax(history['test_acc'])
        best_test_acc = history['test_acc'][best_test_idx]
        ax2.annotate(f'Best: {best_test_acc:.4f}', 
                    xy=(best_test_idx + 1, best_test_acc),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Generalization gap plot
        ax3 = fig.add_subplot(gs[2])
        gap = np.array(history['train_acc']) - np.array(history['test_acc'])
        ax3.plot(epochs, gap, linewidth=2.5, color='#9b59b6', 
                marker='D', markersize=4, markevery=max(1, len(epochs)//20))
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Gap (Train - Test)', fontsize=12, fontweight='bold')
        ax3.set_title('Generalization Gap', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_facecolor('#f8f9fa')
        
        # Add final gap annotation
        final_gap = gap[-1]
        gap_color = 'orange' if abs(final_gap) > 0.1 else 'lightblue'
        ax3.text(0.98, 0.98, f'Final Gap: {final_gap:.4f}',
                transform=ax3.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.8))
        
        plt.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_confusion_matrix(self, dataloader: DataLoader, num_classes: int, 
                             save_path: str = 'confusion_matrix.png') -> None:
        """Generate and save confusion matrix visualization.

        Args:
            dataloader: DataLoader for the dataset.
            num_classes: Number of classes in the dataset.
            save_path: Path to save the visualization.
        """
        print("\nGenerating confusion matrix...")
        
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                logits, _, _ = self.model(X_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Raw counts
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10, fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_xticks(np.arange(num_classes))
        ax1.set_yticks(np.arange(num_classes))
        
        # Normalized
        im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                ax2.text(j, i, f'{cm_normalized[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] < 0.5 else "black",
                        fontsize=10, fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_xticks(np.arange(num_classes))
        ax2.set_yticks(np.arange(num_classes))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_prototype_heatmap(self, save_path: str = 'prototype_heatmap.png') -> None:
        """Visualize prototype patterns as a heatmap.

        Args:
            save_path: Path to save the visualization.
        """
        print("\nGenerating prototype heatmap...")
        
        prototypes = self.model.prototype.prototypes.detach().cpu().numpy()
        num_prototypes = prototypes.shape[0]
        
        fig, ax = plt.subplots(figsize=(14, max(6, num_prototypes * 0.4)))
        
        im = ax.imshow(prototypes, aspect='auto', cmap='coolwarm', interpolation='nearest')
        ax.set_title('Learned Prototype Patterns', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Feature Dimension', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prototype Index', fontsize=12, fontweight='bold')
        ax.set_yticks(np.arange(num_prototypes))
        ax.set_yticklabels([f'P{i}' for i in range(num_prototypes)])
        
        plt.colorbar(im, ax=ax, label='Activation Value', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_sample_predictions(self, dataloader: DataLoader, num_samples: int = 6,
                               save_path: str = 'sample_predictions.png') -> None:
        """Visualize sample time series with predictions.

        Args:
            dataloader: DataLoader for the dataset.
            num_samples: Number of samples to visualize.
            save_path: Path to save the visualization.
        """
        print(f"\nGenerating {num_samples} sample predictions...")
        
        self.model.eval()
        
        # Get first batch
        X_batch, y_batch = next(iter(dataloader))
        X_batch = X_batch.to(self.device)
        
        with torch.no_grad():
            logits, _, _ = self.model(X_batch)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
        
        # Select samples
        num_samples = min(num_samples, len(X_batch))
        samples = X_batch[:num_samples].cpu().numpy()
        true_labels = y_batch[:num_samples].numpy()
        pred_labels = preds[:num_samples].cpu().numpy()
        pred_probs = probs[:num_samples].cpu().numpy()
        
        # Create subplots
        rows = (num_samples + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 3))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx in range(num_samples):
            ax = axes[idx]
            sample = samples[idx]
            
            # Handle univariate vs multivariate
            if sample.ndim == 1:
                ax.plot(sample, linewidth=2, color='#3498db')
            else:
                for ch in range(sample.shape[0]):
                    ax.plot(sample[ch], linewidth=2, label=f'Ch{ch}', alpha=0.8)
                if sample.shape[0] <= 5:
                    ax.legend(fontsize=9)
            
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            confidence = pred_probs[idx][pred_label]
            
            is_correct = true_label == pred_label
            color = 'green' if is_correct else 'red'
            status = 'OK' if is_correct else 'X'
            
            ax.set_title(f'{status} True: {true_label} | Pred: {pred_label} ({confidence:.2%})',
                        fontsize=11, fontweight='bold', color=color)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
        
        # Hide extra subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Time Series Predictions', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")
