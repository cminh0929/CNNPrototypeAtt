"""Test script to verify all enhancements work correctly."""
import torch
import numpy as np
from data.augmentation import TimeSeriesAugmenter
from models.prototype import PrototypeModule
from training.clustering_loss import compute_clustering_loss

print("=" * 70)
print("Testing CNN Prototype Enhancements")
print("=" * 70)

# Test 1: Data Augmentation
print("\n[1/3] Testing Data Augmentation...")
augmenter = TimeSeriesAugmenter(
    jitter_std=0.03,
    scaling_range=(0.8, 1.2),
    time_warp_strength=0.2,
    window_slice_ratio=0.9,
    rotation_prob=0.5,
    augment_prob=1.0  # Always augment for testing
)

# Test univariate
x_univariate = np.random.randn(100).astype(np.float32)
x_aug = augmenter.augment(x_univariate, is_multivariate=False)
assert x_aug.shape == x_univariate.shape, "Univariate augmentation shape mismatch"
print(f"  [OK] Univariate augmentation: {x_univariate.shape} -> {x_aug.shape}")

# Test multivariate
x_multivariate = np.random.randn(3, 100).astype(np.float32)
x_aug = augmenter.augment(x_multivariate, is_multivariate=True)
assert x_aug.shape == x_multivariate.shape, "Multivariate augmentation shape mismatch"
print(f"  [OK] Multivariate augmentation: {x_multivariate.shape} -> {x_aug.shape}")

# Test batch augmentation
X_batch = np.random.randn(16, 100).astype(np.float32)
X_aug_batch = augmenter.augment_batch(X_batch, is_multivariate=False)
assert X_aug_batch.shape == X_batch.shape, "Batch augmentation shape mismatch"
print(f"  [OK] Batch augmentation: {X_batch.shape} -> {X_aug_batch.shape}")

# Test 2: Projection Layer
print("\n[2/3] Testing Projection Layer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Without projection
proto_no_proj = PrototypeModule(
    feature_dim=256,
    num_prototypes=10,
    temperature=0.5,
    use_projection=False
).to(device)

features = torch.randn(32, 256).to(device)
attended, attn = proto_no_proj(features)
assert attended.shape == (32, 256), "No projection output shape mismatch"
assert attn.shape == (32, 10), "Attention weights shape mismatch"
print(f"  [OK] Without projection: features {features.shape} -> attended {attended.shape}")

# With projection (same dimension)
proto_with_proj = PrototypeModule(
    feature_dim=256,
    num_prototypes=10,
    temperature=0.5,
    use_projection=True,
    projection_dim=256
).to(device)

attended, attn = proto_with_proj(features)
assert attended.shape == (32, 256), "With projection output shape mismatch"
assert attn.shape == (32, 10), "Attention weights shape mismatch"
print(f"  [OK] With projection (same dim): features {features.shape} -> attended {attended.shape}")

# With projection (different dimension)
proto_with_proj_diff = PrototypeModule(
    feature_dim=256,
    num_prototypes=10,
    temperature=0.5,
    use_projection=True,
    projection_dim=128
).to(device)

attended, attn = proto_with_proj_diff(features)
assert attended.shape == (32, 256), "With projection (diff dim) output shape mismatch"
assert attn.shape == (32, 10), "Attention weights shape mismatch"
print(f"  [OK] With projection (128 dim): features {features.shape} -> attended {attended.shape}")

# Test 3: Clustering Loss
print("\n[3/3] Testing Clustering Loss...")
features = torch.randn(32, 256, requires_grad=True).to(device)
prototypes = torch.randn(10, 256, requires_grad=True).to(device)
attention_weights = torch.softmax(torch.randn(32, 10), dim=1).to(device)

total_loss, compactness, separation = compute_clustering_loss(
    features,
    prototypes,
    attention_weights,
    compactness_weight=1.0,
    separation_weight=0.5
)

assert total_loss.ndim == 0, "Total loss should be scalar"
assert compactness.ndim == 0, "Compactness loss should be scalar"
assert separation.ndim == 0, "Separation loss should be scalar"
print(f"  [OK] Clustering loss computed successfully")
print(f"    - Total loss: {total_loss.item():.4f}")
print(f"    - Compactness: {compactness.item():.4f}")
print(f"    - Separation: {separation.item():.4f}")

# Test gradient flow
total_loss.backward()
print(f"  [OK] Gradient flow verified")

print("\n" + "=" * 70)
print("All tests passed!")
print("=" * 70)
