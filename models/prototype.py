import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PrototypeModule(nn.Module):
    """Prototype learning module with attention mechanism."""

    def __init__(self, feature_dim: int, num_prototypes: int, temperature: float = 1.0) -> None:
        """Initialize the PrototypeModule.

        Args:
            feature_dim: Dimension of input features.
            num_prototypes: Number of prototypes to learn.
            temperature: Temperature parameter for softmax attention.
        """
        super().__init__()

        self.temperature: float = temperature
        self.prototypes: nn.Parameter = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.1)

        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the prototype module.

        Args:
            features: Input features of shape (batch_size, feature_dim).

        Returns:
            Tuple containing attended features and attention weights.
        """
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        similarity = features_norm @ prototypes_norm.T

        attn = F.softmax(similarity / self.temperature, dim=1)
        attended = attn @ self.prototypes

        return features + attended, attn

    def get_diversity_loss(self) -> torch.Tensor:
        """Calculate diversity loss to encourage prototype separation.

        Returns:
            Diversity loss value.
        """
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        similarity_matrix = prototypes_norm @ prototypes_norm.T
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        return similarity_matrix.abs().mean()
