"""
TSML Module for CNNProto Evaluation and Benchmarking.

This module contains all functionality related to time series model evaluation,
benchmarking against SOTA methods, and comparison table generation.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.cnn_proto_attention import CNNProtoAttentionModel
from utils.seed import set_seed
from utils.device import get_device


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
        seed: int = 42
    ):
        """Initialize CNNProto classifier.

        Args:
            num_prototypes: Number of prototypes (None = num_classes * 2).
            dropout: Dropout rate.
            temperature: Temperature for attention.
            batch_size: Batch size for training.
            epochs: Maximum number of epochs.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            diversity_weight: Weight for diversity loss.
            early_stopping_patience: Patience for early stopping.
            device: Device to use ('auto', 'cuda', or 'cpu').
            seed: Random seed.
        """
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

        self.model = None
        self.device = None
        self.num_classes = None
        self.input_channels = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'CNNProtoClassifier':
        """Fit the model to training data.

        Args:
            X_train: Training data of shape (n_samples, n_timesteps) or (n_samples, n_channels, n_timesteps).
            y_train: Training labels of shape (n_samples,).
            X_val: Optional validation data. If None, uses training data for validation.
            y_val: Optional validation labels.

        Returns:
            Self.
        """
        set_seed(self.seed)
        self.device = get_device(self.device_str)

        # Determine input shape
        if X_train.ndim == 2:
            # Univariate: (n_samples, n_timesteps) -> (n_samples, 1, n_timesteps)
            X_train = X_train[:, np.newaxis, :]
            self.input_channels = 1
        else:
            # Multivariate: (n_samples, n_channels, n_timesteps)
            self.input_channels = X_train.shape[1]

        # Determine number of classes
        self.num_classes = len(np.unique(y_train))

        # Determine number of prototypes
        num_prototypes = self.num_prototypes
        if num_prototypes is None:
            num_prototypes = self.num_classes * 2

        # Create model
        self.model = CNNProtoAttentionModel(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            num_prototypes=num_prototypes,
            dropout=self.dropout,
            temperature=self.temperature
        ).to(self.device)

        # Create training data loader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Create validation data loader
        if X_val is not None and y_val is not None:
            # Handle validation data shape
            if X_val.ndim == 2:
                X_val = X_val[:, np.newaxis, :]
            
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            # Use training data as validation if no validation data provided
            val_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Train model
        from training.trainer import Trainer
        trainer = Trainer(
            model=self.model,
            device=self.device,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            diversity_weight=self.diversity_weight
        )

        trainer.train(
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_patience
        )

        # Load best model
        self.model.load_state_dict(trainer.best_model_state)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test data.

        Args:
            X_test: Test data of shape (n_samples, n_timesteps) or (n_samples, n_channels, n_timesteps).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet!")

        # Handle input shape
        if X_test.ndim == 2:
            X_test = X_test[:, np.newaxis, :]

        # Create data loader
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Predict
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                logits, _, _ = self.model(X_batch)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities for test data.

        Args:
            X_test: Test data of shape (n_samples, n_timesteps) or (n_samples, n_channels, n_timesteps).

        Returns:
            Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet!")

        # Handle input shape
        if X_test.ndim == 2:
            X_test = X_test[:, np.newaxis, :]

        # Create data loader
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Predict probabilities
        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                logits, _, _ = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)
