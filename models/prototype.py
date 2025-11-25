import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrototypeModule(nn.Module):
    """Prototype learning module with attention mechanism and optional projection."""

    def __init__(
        self,
        feature_dim: int,
        num_prototypes: int,
        temperature: float = 1.0,
        projection_dim: Optional[int] = None,
        use_projection: bool = False
    ) -> None:
        """Initialize the PrototypeModule.

        Args:
            feature_dim: Dimension of input features.
            num_prototypes: Number of prototypes to learn.
            temperature: Temperature parameter for softmax attention.
            projection_dim: Dimension of projection space. If None, uses feature_dim.
            use_projection: Whether to use projection layer.
        """
        super().__init__()

        self.temperature: float = temperature
        self.use_projection: bool = use_projection
        self.feature_dim: int = feature_dim
        
        # Determine projection dimension
        if projection_dim is None:
            projection_dim = feature_dim
        self.projection_dim: int = projection_dim
        
        # Add projection layer if enabled
        if self.use_projection:
            self.projection: nn.Linear = nn.Linear(feature_dim, projection_dim)
            # Add inverse projection to map back to feature space
            self.inverse_projection: nn.Linear = nn.Linear(projection_dim, feature_dim)
            self.prototypes: nn.Parameter = nn.Parameter(
                torch.randn(num_prototypes, projection_dim) * 0.1
            )
        else:
            self.projection: Optional[nn.Linear] = None
            self.inverse_projection: Optional[nn.Linear] = None
            self.prototypes: nn.Parameter = nn.Parameter(
                torch.randn(num_prototypes, feature_dim) * 0.1
            )

        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the prototype module.

        Args:
            features: Input features of shape (batch_size, feature_dim).

        Returns:
            Tuple containing attended features and attention weights.
        """
        # Apply projection if enabled
        if self.use_projection:
            # Project features to prototype space
            projected_features = self.projection(features)
            projected_features = F.normalize(projected_features, dim=1)
            prototypes_norm = F.normalize(self.prototypes, dim=1)
            similarity = projected_features @ prototypes_norm.T
            
            # Compute attention and attended features in projection space
            attn = F.softmax(similarity / self.temperature, dim=1)
            attended_projected = attn @ self.prototypes
            
            # Project back to original feature space
            attended_features = self.inverse_projection(attended_projected)
            
            # Return features + attended in original space
            return features + attended_features, attn
        else:
            # Original implementation without projection
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
