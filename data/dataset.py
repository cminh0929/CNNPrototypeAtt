from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray

from data.augmentation import TimeSeriesAugmenter


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""

    def __init__(self, X: NDArray[np.float32], y: NDArray[np.int_]) -> None:
        """Initialize the TimeSeriesDataset.

        Args:
            X: Input time series data.
            y: Target labels.
        """
        self.X: torch.Tensor = torch.FloatTensor(X)
        self.y: torch.Tensor = torch.LongTensor(y)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample and its label.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the sample and its label.
        """
        return self.X[idx], self.y[idx]


class AugmentedTimeSeriesDataset(Dataset):
    """PyTorch Dataset with on-the-fly data augmentation."""

    def __init__(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int_],
        augmenter: TimeSeriesAugmenter,
        is_multivariate: bool = False
    ) -> None:
        """Initialize the AugmentedTimeSeriesDataset.

        Args:
            X: Input time series data.
            y: Target labels.
            augmenter: TimeSeriesAugmenter instance for augmentation.
            is_multivariate: Whether the time series is multivariate.
        """
        self.X: NDArray[np.float32] = X
        self.y: NDArray[np.int_] = y
        self.augmenter: TimeSeriesAugmenter = augmenter
        self.is_multivariate: bool = is_multivariate

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an augmented sample and its label.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the augmented sample and its label.
        """
        x = self.X[idx]
        y = self.y[idx]
        
        # Apply augmentation
        x_aug = self.augmenter.augment(x, self.is_multivariate)
        
        return torch.FloatTensor(x_aug), torch.LongTensor([y]).squeeze()
