from typing import Dict, Any, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader
import os

from data.dataset import TimeSeriesDataset, AugmentedTimeSeriesDataset
from data.augmentation import TimeSeriesAugmenter


class DataLoaderManager:
    """Manages data loading and preprocessing for time series datasets."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        datasets_dir: str = "datasets",
        use_augmentation: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the DataLoaderManager.

        Args:
            dataset_name: Name of the UCR/UEA dataset to load.
            batch_size: Number of samples per batch.
            datasets_dir: Directory containing the local datasets.
            use_augmentation: Whether to apply data augmentation to training data.
            augmentation_config: Configuration dictionary for augmentation parameters.
        """
        self.dataset_name: str = dataset_name
        self.batch_size: int = batch_size
        self.datasets_dir: str = datasets_dir
        self.use_augmentation: bool = use_augmentation
        self.augmentation_config: Dict[str, Any] = augmentation_config or {}

        self.X_train: Optional[NDArray[np.float32]] = None
        self.y_train: Optional[NDArray[np.int_]] = None
        self.X_test: Optional[NDArray[np.float32]] = None
        self.y_test: Optional[NDArray[np.int_]] = None
        self.num_classes: Optional[int] = None
        self.input_channels: Optional[int] = None
        self.time_steps: Optional[int] = None

        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def _load_tsv_file(self, filepath: str) -> Tuple[NDArray[np.float32], NDArray[np.int_]]:
        """Load data from a TSV file.

        Args:
            filepath: Path to the TSV file.

        Returns:
            Tuple of (X, y) where X is the time series data and y is the labels.
        """
        data = np.loadtxt(filepath, delimiter='\t')
        y = data[:, 0].astype(int)
        X = data[:, 1:]
        return X, y

    def load_and_prepare(self) -> "DataLoaderManager":
        """Load dataset and prepare data loaders with normalization.

        Returns:
            Self for method chaining.
        """
        print("\nData Loading: {}".format(self.dataset_name))
        print("-" * 70)

        # Construct file paths
        dataset_path = os.path.join(self.datasets_dir, self.dataset_name)
        train_file = os.path.join(dataset_path, f"{self.dataset_name}_TRAIN.tsv")
        test_file = os.path.join(dataset_path, f"{self.dataset_name}_TEST.tsv")

        # Check if files exist
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        print(f"Loading from local files:")
        print(f"  Train: {train_file}")
        print(f"  Test:  {test_file}")

        # Load data from TSV files
        X_train, y_train = self._load_tsv_file(train_file)
        X_test, y_test = self._load_tsv_file(test_file)

        print(f"Raw shape: {X_train.shape}")

        # Handle multivariate vs univariate
        if X_train.ndim == 2:
            # Univariate case: shape is (N, L)
            self.input_channels = 1
            print(f"UNIVARIATE")
        elif X_train.ndim == 3:
            # Multivariate case: shape should be (N, C, L)
            if X_train.shape[2] <= X_train.shape[1]:
                X_train = np.transpose(X_train, (0, 2, 1))
                X_test = np.transpose(X_test, (0, 2, 1))
                print(f"Transposed to: {X_train.shape} -> (N, C, L)")

            self.input_channels = X_train.shape[1]
            if self.input_channels > 1:
                print(f"MULTIVARIATE DETECTED: {self.input_channels} channels")
            else:
                X_train = np.squeeze(X_train, axis=1)
                X_test = np.squeeze(X_test, axis=1)
                self.input_channels = 1
                print(f"UNIVARIATE (squeezed)")

        print("Applying per-channel z-score normalization...", end="")
        def instance_norm(X):
            # X shape: (N, C, L) hoặc (N, L)
            # Tính mean/std dọc theo trục thời gian (trục cuối cùng)
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True) + 1e-8 # Cộng epsilon để tránh chia cho 0
            return (X - mean) / std

        # Áp dụng
        X_train = instance_norm(X_train)
        X_test = instance_norm(X_test)
        
        print(" done (Instance Norm)")

        # Map labels to 0-indexed integers
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_train = np.array([label_map[label] for label in y_train])
        y_test = np.array([label_map[label] for label in y_test])

        self.num_classes = len(unique_labels)
        self.time_steps = X_train.shape[-1]

        print(f"Classes: {self.num_classes}")
        print(f"Time steps: {self.time_steps}")
        print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train
        self.y_test = y_test

        # Create data loaders with optional augmentation
        if self.use_augmentation:
            print("Data augmentation: ENABLED")
            augmenter = TimeSeriesAugmenter(**self.augmentation_config)
            train_dataset = AugmentedTimeSeriesDataset(
                self.X_train,
                self.y_train,
                augmenter,
                is_multivariate=(self.input_channels > 1)
            )
        else:
            print("Data augmentation: DISABLED")
            train_dataset = TimeSeriesDataset(self.X_train, self.y_train)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        self.test_loader = DataLoader(
            TimeSeriesDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            shuffle=False
        )

        print("-" * 70)
        return self

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Return train and test data loaders.

        Returns:
            Tuple containing train and test DataLoader objects.
        """
        return self.train_loader, self.test_loader

    def get_info(self) -> Dict[str, int]:
        """Return dataset information.

        Returns:
            Dictionary containing input channels, number of classes, and time steps.
        """
        return {
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'time_steps': self.time_steps
        }
