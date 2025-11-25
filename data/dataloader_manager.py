from typing import Dict, Any, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader
import os
from scipy.io import arff
import pandas as pd

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

    def _load_txt_file(self, filepath: str) -> Tuple[NDArray[np.float32], NDArray[np.int_]]:
        """Load data from a TXT file (space-delimited, label in first column).

        Args:
            filepath: Path to the TXT file.

        Returns:
            Tuple of (X, y) where X is the time series data and y is the labels.
        """
        data = np.loadtxt(filepath, delimiter=' ')
        y = data[:, 0].astype(int)
        X = data[:, 1:]
        return X, y

    def _load_ts_file(self, filepath: str) -> Tuple[NDArray[np.float32], NDArray[np.int_]]:
        """Load data from a TS file (comma-delimited, label after colon, with comments).
        
        Supports both univariate and multivariate time series:
        - Univariate: values:label
        - Multivariate: dim1_values:dim2_values:...:label

        Args:
            filepath: Path to the TS file.

        Returns:
            Tuple of (X, y) where X is the time series data and y is the labels.
            For multivariate, X shape is (N, C, L) where C is number of channels.
        """
        X_list = []
        y_list = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#') or line.startswith('@') or line.startswith('%'):
                    continue
                
                # Split by colon - last part is label, rest are dimensions
                if ':' in line:
                    parts = line.split(':')
                    label_str = parts[-1].strip()
                    
                    # Try to parse as int, if fails keep as string
                    try:
                        label = int(label_str)
                    except ValueError:
                        label = label_str
                    
                    # All parts except the last are data dimensions
                    dimensions = []
                    for dim_str in parts[:-1]:
                        # Parse comma-separated values for this dimension
                        values = np.array([float(x) for x in dim_str.split(',')], dtype=np.float32)
                        dimensions.append(values)
                    
                    # Stack dimensions: if univariate, shape is (L,); if multivariate, shape is (C, L)
                    if len(dimensions) == 1:
                        sample = dimensions[0]  # Univariate: (L,)
                    else:
                        sample = np.stack(dimensions, axis=0)  # Multivariate: (C, L)
                    
                    X_list.append(sample)
                    y_list.append(label)
        
        # Convert to numpy array
        if len(X_list) > 0:
            # Check if univariate or multivariate
            if X_list[0].ndim == 1:
                # Univariate: (N, L)
                X = np.array(X_list, dtype=np.float32)
            else:
                # Multivariate: (N, C, L)
                X = np.array(X_list, dtype=np.float32)
        else:
            X = np.array([], dtype=np.float32)
        
        # Handle string labels (convert to integers)
        y_array = np.array(y_list)
        if y_array.dtype == object or y_array.dtype.kind in ['S', 'U', 'O']:
            unique_labels = np.unique(y_array)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y_array], dtype=int)
        else:
            y = y_array.astype(int)
        
        return X, y

    def _load_csv_file(self, filepath: str) -> Tuple[NDArray[np.float32], NDArray[np.int_]]:
        """Load data from a CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            Tuple of (X, y) where X is the time series data and y is the labels.
        """
        data = np.loadtxt(filepath, delimiter=',')
        y = data[:, 0].astype(int)
        X = data[:, 1:]
        return X, y

    def _load_arff_file(self, filepath: str) -> Tuple[NDArray[np.float32], NDArray[np.int_]]:
        """Load data from an ARFF file.

        Args:
            filepath: Path to the ARFF file.

        Returns:
            Tuple of (X, y) where X is the time series data and y is the labels.
        """
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        
        # The last column is typically the class label in ARFF files
        y = df.iloc[:, -1].values
        
        # Convert labels to integers if they are strings/bytes
        if y.dtype == object or y.dtype.kind in ['S', 'U', 'O']:
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y], dtype=int)
        else:
            y = y.astype(int)
        
        # All other columns are features
        X = df.iloc[:, :-1].values.astype(np.float32)
        
        return X, y

    def _detect_and_load_file(self, filepath: str) -> Tuple[NDArray[np.float32], NDArray[np.int_]]:
        """Detect file format and load data using the appropriate method.

        Args:
            filepath: Path to the data file.

        Returns:
            Tuple of (X, y) where X is the time series data and y is the labels.
        
        Raises:
            ValueError: If file format is not supported.
        """
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext == '.tsv':
            return self._load_tsv_file(filepath)
        elif ext == '.ts':
            return self._load_ts_file(filepath)
        elif ext == '.txt':
            return self._load_txt_file(filepath)
        elif ext == '.csv':
            return self._load_csv_file(filepath)
        elif ext == '.arff':
            return self._load_arff_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: .tsv, .ts, .txt, .csv, .arff")

    def load_and_prepare(self) -> "DataLoaderManager":
        """Load dataset and prepare data loaders with normalization.

        Returns:
            Self for method chaining.
        """
        print("\nData Loading: {}".format(self.dataset_name))
        print("-" * 70)

        # Construct dataset path
        dataset_path = os.path.join(self.datasets_dir, self.dataset_name)
        
        # Try to find files with supported extensions
        supported_extensions = ['.tsv', '.ts', '.txt', '.csv', '.arff']
        train_file = None
        test_file = None
        
        for ext in supported_extensions:
            train_candidate = os.path.join(dataset_path, f"{self.dataset_name}_TRAIN{ext}")
            test_candidate = os.path.join(dataset_path, f"{self.dataset_name}_TEST{ext}")
            
            if os.path.exists(train_candidate) and os.path.exists(test_candidate):
                train_file = train_candidate
                test_file = test_candidate
                print(f"Detected file format: {ext.upper()}")
                break
        
        # Check if files were found
        if train_file is None or test_file is None:
            raise FileNotFoundError(
                f"Dataset files not found for '{self.dataset_name}' in '{dataset_path}'.\n"
                f"Expected files with extensions: {', '.join(supported_extensions)}\n"
                f"Looking for: {self.dataset_name}_TRAIN.<ext> and {self.dataset_name}_TEST.<ext>"
            )

        print(f"Loading from local files:")
        print(f"  Train: {train_file}")
        print(f"  Test:  {test_file}")

        # Load data using auto-detection
        X_train, y_train = self._detect_and_load_file(train_file)
        X_test, y_test = self._detect_and_load_file(test_file)

        print(f"Raw shape: {X_train.shape}")

        # Handle multivariate vs univariate
        if X_train.ndim == 2:
            # Univariate case: shape is (N, L)
            self.input_channels = 1
            print(f"UNIVARIATE")
        elif X_train.ndim == 3:
            # Multivariate case: shape should be (N, C, L)
            # Check if transpose is needed based on shape heuristic
            if X_train.shape[2] <= X_train.shape[1]:
                print(f"\n[WARNING] Auto-transpose detected!")
                print(f"  Original shape: {X_train.shape}")
                print(f"  Assumption: dim1={X_train.shape[1]} is length, dim2={X_train.shape[2]} is channels")
                print(f"  This assumes #channels <= #timesteps")
                X_train = np.transpose(X_train, (0, 2, 1))
                X_test = np.transpose(X_test, (0, 2, 1))
                print(f"  Transposed to: {X_train.shape} -> (N, C, L)")
                print(f"  If this is incorrect, please format your data as (N, C, L)\n")

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
            # X shape: (N, C, L) or (N, L)
            # Compute mean/std along time dimension (last axis)
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
            return (X - mean) / std

        # Apply instance normalization
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
