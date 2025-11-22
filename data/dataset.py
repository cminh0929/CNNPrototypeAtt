from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray


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
