import torch
import torch.nn.functional as F
from typing import Tuple


def compute_clustering_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    attention_weights: torch.Tensor,
    compactness_weight: float = 1.0,
    separation_weight: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute clustering-based loss for prototype learning.

    Args:
        features: Feature representations of shape (batch_size, feature_dim).
        prototypes: Prototype vectors of shape (num_prototypes, feature_dim).
        attention_weights: Attention weights of shape (batch_size, num_prototypes).
        compactness_weight: Weight for intra-cluster compactness loss.
        separation_weight: Weight for inter-cluster separation loss.

    Returns:
        Tuple containing (total_clustering_loss, compactness_loss, separation_loss).
    """
    # Normalize features and prototypes
    features_norm = F.normalize(features, dim=1)
    prototypes_norm = F.normalize(prototypes, dim=1)
    
    # Compute compactness loss (intra-cluster)
    compactness_loss = compute_compactness_loss(features_norm, prototypes_norm, attention_weights)
    
    # Compute separation loss (inter-cluster)
    separation_loss = compute_separation_loss(prototypes_norm)
    
    # Combine losses
    total_loss = compactness_weight * compactness_loss - separation_weight * separation_loss
    
    return total_loss, compactness_loss, separation_loss


def compute_compactness_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    attention_weights: torch.Tensor
) -> torch.Tensor:
    """Compute intra-cluster compactness loss.

    This loss minimizes the distance between samples and their assigned prototypes,
    encouraging tight clusters.

    Args:
        features: Normalized feature representations of shape (batch_size, feature_dim).
        prototypes: Normalized prototype vectors of shape (num_prototypes, feature_dim).
        attention_weights: Attention weights of shape (batch_size, num_prototypes).

    Returns:
        Compactness loss value.
    """
    # Compute pairwise distances between features and prototypes
    # Distance = 1 - cosine_similarity (since vectors are normalized)
    similarity = features @ prototypes.T  # (batch_size, num_prototypes)
    distances = 1.0 - similarity
    
    # Weighted average distance (using attention weights as soft assignments)
    weighted_distances = distances * attention_weights
    compactness_loss = weighted_distances.sum(dim=1).mean()
    
    return compactness_loss


def compute_separation_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """Compute inter-cluster separation loss.

    This loss maximizes the distance between different prototypes,
    encouraging well-separated clusters.

    Args:
        prototypes: Normalized prototype vectors of shape (num_prototypes, feature_dim).

    Returns:
        Separation loss value (negative, to be maximized).
    """
    # Compute pairwise similarities between prototypes
    similarity_matrix = prototypes @ prototypes.T  # (num_prototypes, num_prototypes)
    
    # Mask out diagonal (self-similarity)
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    similarity_matrix = similarity_matrix * (1 - mask)
    
    # Average similarity between different prototypes
    # We want to minimize this (maximize separation)
    num_pairs = similarity_matrix.size(0) * (similarity_matrix.size(0) - 1)
    separation_loss = similarity_matrix.sum() / num_pairs
    
    return separation_loss
