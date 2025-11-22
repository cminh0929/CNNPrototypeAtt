import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for extracting features from time series data."""

    def __init__(self, input_channels: int, dropout: float = 0.1) -> None:
        """Initialize the CNNFeatureExtractor.

        Args:
            input_channels: Number of input channels.
            dropout: Dropout probability.
        """
        super().__init__()

        self.conv1: nn.Conv1d = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.dropout1: nn.Dropout = nn.Dropout(dropout)

        self.conv2: nn.Conv1d = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.dropout2: nn.Dropout = nn.Dropout(dropout)

        self.conv3: nn.Conv1d = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.dropout3: nn.Dropout = nn.Dropout(dropout)

        self.feature_dim: int = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN feature extractor.

        Args:
            x: Input tensor of shape (batch_size, channels, length) or (batch_size, length).

        Returns:
            Extracted features of shape (batch_size, feature_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool1d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)
