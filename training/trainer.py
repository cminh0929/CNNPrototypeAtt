import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional, Any
import time
import copy

from training.clustering_loss import compute_clustering_loss


class Trainer:
    """Handles model training with optimization and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        diversity_weight: float = 0.01,
        label_smoothing: float = 0.0,
        clustering_weight: float = 0.0,
        clustering_config: Optional[Dict[str, float]] = None
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: PyTorch model to train.
            device: Device to train on.
            lr: Learning rate.
            weight_decay: Weight decay for regularization.
            diversity_weight: Weight for prototype diversity loss.
            label_smoothing: Label smoothing factor.
            clustering_weight: Weight for clustering loss.
            clustering_config: Configuration for clustering loss (compactness/separation weights).
        """
        self.model: nn.Module = model
        self.device: torch.device = device
        self.diversity_weight: float = diversity_weight
        self.clustering_weight: float = clustering_weight
        self.clustering_config: Dict[str, float] = clustering_config or {
            'compactness_weight': 1.0,
            'separation_weight': 0.5
        }

        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer: Adam = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler: ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )

        self.best_model_state: Optional[Dict[str, Any]] = None
        self.best_acc: float = 0.0
        self.best_epoch: int = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Tuple containing average loss and training accuracy.
        """
        self.model.train()
        total_loss = 0.0
        preds: List[int] = []
        labels: List[int] = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            logits, attn, features = self.model(X_batch)
            ce_loss = self.criterion(logits, y_batch)

            # Diversity loss
            div_loss = self.model.prototype.get_diversity_loss()
            
            # Clustering loss
            if self.clustering_weight > 0:
                clustering_loss, _, _ = compute_clustering_loss(
                    features,
                    self.model.prototype.prototypes,
                    attn,
                    compactness_weight=self.clustering_config['compactness_weight'],
                    separation_weight=self.clustering_config['separation_weight']
                )
                loss = ce_loss + self.diversity_weight * div_loss + self.clustering_weight * clustering_loss
            else:
                loss = ce_loss + self.diversity_weight * div_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(labels, preds)

        return avg_loss, train_acc

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: Optional[int] = 20
    ) -> Tuple[Dict[str, List[float]], int]:
        """Execute the full training loop.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.
            epochs: Number of training epochs.
            early_stopping_patience: Patience for early stopping.

        Returns:
            Tuple containing training history dictionary and best epoch number.
        """
        print("\nTraining")
        print("-" * 70)

        history: Dict[str, List[float]] = {
            'loss': [],
            'train_acc': [],
            'test_acc': [],
            'learning_rate': []
        }
        start_time = time.time()
        best_test_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            loss, train_acc = self.train_epoch(train_loader)
            test_acc = self._evaluate_epoch(test_loader)

            self.scheduler.step(test_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            history['loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['learning_rate'].append(current_lr)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                self.best_epoch = epoch + 1  # Store the best epoch (1-indexed)
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best test accuracy: {best_test_acc:.4f}")
                self.model.load_state_dict(self.best_model_state)
                break

        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f}s")
        print(f"Best test accuracy: {best_test_acc:.4f}")
        print("-" * 70)

        return history, self.best_epoch

    def _evaluate_epoch(self, dataloader: DataLoader) -> float:
        """Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data.

        Returns:
            Accuracy score.
        """
        self.model.eval()
        preds: List[int] = []
        labels: List[int] = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits, _, _ = self.model(X_batch)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y_batch.cpu().numpy())

        return accuracy_score(labels, preds)
    
    def initialize_prototypes(self, train_loader: DataLoader):
        """Wrapper to call model's prototype initialization."""
        if hasattr(self.model, 'initialize_prototypes'):
            print("\n[Trainer] Initializing prototypes via K-Means...")
            self.model.initialize_prototypes(train_loader, device=self.device)
        else:
            print("\n[Warning] Model does not support K-Means initialization.")