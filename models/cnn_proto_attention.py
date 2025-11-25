import torch
import torch.nn as nn
from typing import Optional, Tuple
from sklearn.cluster import KMeans 
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
        temperature: float = 1.0,
        use_projection: bool = False,
        projection_dim: Optional[int] = None
    ) -> None:
        """Initialize the CNNProtoAttentionModel.

        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            num_prototypes: Number of prototypes. Defaults to num_classes * 3.
            dropout: Dropout probability.
            temperature: Temperature parameter for prototype attention.
            use_projection: Whether to use projection layer in prototype module.
            projection_dim: Dimension of projection space. If None, uses feature_dim.
        """
        super().__init__()

        if num_prototypes is None:
            num_prototypes = num_classes * 3

        self.cnn: CNNFeatureExtractor = CNNFeatureExtractor(input_channels, dropout=dropout)
        self.prototype: PrototypeModule = PrototypeModule(
            self.cnn.feature_dim,
            num_prototypes,
            temperature=temperature,
            use_projection=use_projection,
            projection_dim=projection_dim
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
    
    def initialize_prototypes(self, data_loader, device='cuda'):
        """Initialize Prototypes using K-Means instead of random initialization."""
        print("Initializing prototypes with K-Means...")
        self.eval()
        all_features = []
        
        # 1. Extract features from entire training set
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                features = self.cnn(x)
                
                # If using projection, project features first
                if self.prototype.use_projection:
                    features = self.prototype.projection(features)
                
                all_features.append(features.cpu())
        
        all_features = torch.cat(all_features, dim=0).numpy()
        
        # 2. Run K-Means to find cluster centers
        kmeans = KMeans(n_clusters=self.prototype.prototypes.shape[0], n_init=10)
        kmeans.fit(all_features)
        
        # 3. Assign cluster centers to model prototypes
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        self.prototype.prototypes.data = torch.nn.functional.normalize(cluster_centers, dim=1)
        print("Prototypes initialized!")