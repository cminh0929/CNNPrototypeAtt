"""
Evaluation script for CNNProto model with standardized result format.
UPDATED: Includes K-Means Initialization, Adaptive Batch Sizing, and new features support.
"""

from typing import Dict, Any, Optional
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
import json

from models.cnn_proto_attention import CNNProtoAttentionModel
from data.dataloader_manager import DataLoaderManager
from config.config_manager import ConfigManager
from utils.seed import set_seed
from utils.device import get_device
from utils.dataset_utils import discover_datasets, validate_dataset, list_datasets


class CNNProtoClassifier:
    """Wrapper class to make CNNProto compatible with sklearn-style interface."""

    def __init__(
        self,
        num_prototypes: Optional[int] = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        batch_size: int = 32,
        epochs: int = 200,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        diversity_weight: float = 0.01,
        early_stopping_patience: int = 20,
        device: str = 'auto',
        seed: int = 42,
        use_projection: bool = False,
        projection_dim: Optional[int] = None,
        clustering_weight: float = 0.0,
        clustering_config: Optional[Dict[str, float]] = None
    ):
        self.num_prototypes = num_prototypes
        self.dropout = dropout
        self.temperature = temperature
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.diversity_weight = diversity_weight
        self.early_stopping_patience = early_stopping_patience
        self.device_str = device
        self.seed = seed
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.clustering_weight = clustering_weight
        self.clustering_config = clustering_config or {}

        self.model = None
        self.device = None
        self.num_classes = None
        self.input_channels = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'CNNProtoClassifier':
        """Fit the model to training data."""
        set_seed(self.seed)
        self.device = get_device(self.device_str)

        # 1. Determine input shape
        if X_train.ndim == 2:
            # Univariate: (n_samples, n_timesteps) -> (n_samples, 1, n_timesteps)
            X_train = X_train[:, np.newaxis, :]
            self.input_channels = 1
        else:
            # Multivariate: (n_samples, n_channels, n_timesteps)
            self.input_channels = X_train.shape[1]

        # 2. Determine number of classes
        self.num_classes = len(np.unique(y_train))

        # 3. --- ADAPTIVE CONFIGURATION (QUAN TRỌNG) ---
        # Tự động điều chỉnh Batch Size cho tập dữ liệu nhỏ (Meat, Coffee...)
        # Nếu không có đoạn này, benchmark sẽ bị thấp do batch size 32 quá lớn.
        num_samples = len(X_train)
        adaptive_batch_size = min(self.batch_size, int(num_samples / 4))
        adaptive_batch_size = max(4, adaptive_batch_size) # Min là 4
        
        if adaptive_batch_size != self.batch_size:
            print(f"  [Adaptive] Adjusting batch size: {self.batch_size} -> {adaptive_batch_size} (Samples: {num_samples})")

        # Tự động tính số Prototypes
        num_prototypes = self.num_prototypes
        if num_prototypes is None:
            num_prototypes = self.num_classes * 3
            # Nếu nhiều class quá (như Adiac 37 class), đảm bảo đủ proto
            num_prototypes = max(num_prototypes, 16)

        # 4. Create model with new features support
        self.model = CNNProtoAttentionModel(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            num_prototypes=num_prototypes,
            dropout=self.dropout,
            temperature=self.temperature,
            use_projection=self.use_projection,
            projection_dim=self.projection_dim
        ).to(self.device)

        # 5. Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=adaptive_batch_size, # Dùng batch size đã thích ứng
            shuffle=True
        )

        # 6. --- K-MEANS INITIALIZATION (BỔ SUNG) ---
        # Đây là đoạn bạn nhắc thiếu trước đó
        if hasattr(self.model, 'initialize_prototypes'):
            print("  [K-Means Init] Initializing prototypes based on training data...")
            self.model.initialize_prototypes(train_loader, device=self.device)
        else:
            print("  [Warning] Model missing 'initialize_prototypes'. Using Random Init.")

        # 7. Train model with new features
        from training.trainer import Trainer
        trainer = Trainer(
            model=self.model,
            device=self.device,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            diversity_weight=self.diversity_weight,
            clustering_weight=self.clustering_weight,
            clustering_config=self.clustering_config
        )

        # Use train loader for validation in this script context
        trainer.train(
            train_loader=train_loader,
            test_loader=train_loader,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_patience
        )

        # Load best model
        if trainer.best_model_state is not None:
            self.model.load_state_dict(trainer.best_model_state)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test data."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet!")

        if X_test.ndim == 2:
            X_test = X_test[:, np.newaxis, :]

        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                logits, _, _ = self.model(X_batch)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)


def evaluate_with_tsml(dataset_name: str, config_manager: ConfigManager) -> None:
    """Evaluate CNNProto and save results in standardized format."""
    print(f"\nEvaluating CNNProto on {dataset_name}")
    print("-" * 70)

    try:
        from sklearn.metrics import accuracy_score
        
        # Load data using DataLoaderManager (Ensure Instance Norm is applied inside here!)
        # Note: Augmentation is disabled for evaluation (only use normalized data)
        data_manager = DataLoaderManager(
            dataset_name=dataset_name,
            batch_size=32,
            use_augmentation=False  # No augmentation during evaluation
        )
        data_manager.load_and_prepare()

        X_train = data_manager.X_train
        y_train = data_manager.y_train
        X_test = data_manager.X_test
        y_test = data_manager.y_test

        # Get configuration
        config = config_manager.get_config(dataset_name, verbose=False)

        # Create and train classifier with new features
        print(f"Training CNNProto classifier (Train size: {len(X_train)})...")
        clf = CNNProtoClassifier(
            num_prototypes=config.get('num_prototypes'),
            dropout=config.get('dropout', 0.1),
            temperature=config.get('temperature', 1.0),
            batch_size=config.get('batch_size', 16),
            epochs=config.get('epochs', 200),
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001),
            diversity_weight=config.get('diversity_weight', 0.05),
            early_stopping_patience=config.get('early_stopping_patience', 50),
            seed=config.get('seed', 42),
            use_projection=config.get('use_projection', False),
            projection_dim=config.get('projection_dim', None),
            clustering_weight=config.get('clustering_weight', 0.0),
            clustering_config=config.get('clustering_loss', {})
        )

        clf.fit(X_train, y_train)

        # Predict
        print("Evaluating on test set...")
        y_pred = clf.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nResults for {dataset_name}:")
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Save results in standardized format
        results_dir = Path("results") / dataset_name / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "dataset": dataset_name,
            "classifier": "CNNProto",
            "accuracy": float(accuracy),
            "num_samples": len(X_train),
            "config": config
        }

        with open(results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_dir / 'results.json'}")
        print("-" * 70)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main entry point for CNNProto evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate CNNProto model")

    parser.add_argument('--dataset', type=str, default='GunPoint')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--list', action='store_true')

    args = parser.parse_args()
    config_manager = ConfigManager("config.yaml")

    if args.list:
        list_datasets(verbose=True)
        sys.exit(0)

    if args.all:
        datasets = discover_datasets()
        if not datasets:
            print("No datasets found!")
            sys.exit(1)
        for i, dataset_name in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] Processing: {dataset_name}")
            evaluate_with_tsml(dataset_name, config_manager)
        sys.exit(0)

    if validate_dataset(args.dataset):
        evaluate_with_tsml(args.dataset, config_manager)
    else:
        print(f"Dataset {args.dataset} not found.")

if __name__ == "__main__":
    main()