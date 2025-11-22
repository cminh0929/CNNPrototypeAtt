import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.cnn_backbone import CNNFeatureExtractor
from models.prototype import PrototypeModule


class CNNProtoAttentionModel(nn.Module):
    """Complete CNN-based prototype attention model for time series classification."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_prototypes: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0
    ) -> None:
        """Initialize the CNNProtoAttentionModel.

        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            num_prototypes: Number of prototypes. Defaults to num_classes * 3.
            dropout: Dropout probability.
            temperature: Temperature parameter for prototype attention.
        """
        super().__init__()

        if num_prototypes is None:
            num_prototypes = num_classes * 3

        self.cnn: CNNFeatureExtractor = CNNFeatureExtractor(input_channels, dropout=dropout)
        self.prototype: PrototypeModule = PrototypeModule(
            self.cnn.feature_dim,
            num_prototypes,
            temperature=temperature
        )
        self.classifier: nn.Linear = nn.Linear(self.cnn.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, length) or (batch_size, length).

        Returns:
            Tuple containing logits, attention weights, and features.
        """
        features = self.cnn(x)
        attended_features, attn = self.prototype(features)
        logits = self.classifier(attended_features)
        return logits, attn, features
