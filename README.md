# CNN Prototype Attention for Time Series Classification

## Overview

A deep learning framework for time series classification that combines convolutional neural networks with prototype-based attention mechanisms. The system learns interpretable prototypes in feature space and uses attention-weighted similarity matching for classification on univariate and multivariate time series datasets.

## Directory Structure

```
CNNProto/
├── config/                          # Configuration management
│   ├── config_manager.py           # YAML config loader and dataset-specific overrides
│   └── __init__.py
│
├── config.yaml                      # Global hyperparameters (learning rate, epochs, etc.)
│
├── data/                            # Data loading and preprocessing pipeline
│   ├── augmentation.py             # Time series augmentation (jitter, scaling, time warp, rotation)
│   ├── dataloader_manager.py       # Multi-format loader (.tsv, .ts, .txt, .csv, .arff)
│   ├── dataset.py                  # PyTorch Dataset classes with optional augmentation
│   └── __init__.py
│
├── models/                          # Neural network architecture components
│   ├── cnn_backbone.py             # 3-layer CNN feature extractor (64→128→256 channels)
│   ├── cnn_proto_attention.py      # Main model integrating CNN + Prototype + Classifier
│   ├── prototype.py                # Prototype learning module with attention mechanism
│   └── __init__.py
│
├── training/                        # Training loop and loss functions
│   ├── clustering_loss.py          # Compactness and separation losses for prototypes
│   ├── evaluator.py                # Model evaluation utilities
│   ├── trainer.py                  # Training loop with early stopping and LR scheduling
│   └── __init__.py
│
├── utils/                           # Utility modules
│   ├── dataset_info.py             # Dataset metadata display
│   ├── dataset_utils.py            # Dataset discovery and validation
│   ├── device.py                   # CUDA/CPU device selection
│   ├── results_manager.py          # Experiment tracking (current/best/history JSON)
│   ├── seed.py                     # Random seed control
│   └── __init__.py
│
├── visualization/                   # Visualization generation
│   ├── visualizer.py               # PCA, confusion matrix, training curves, prototypes
│   └── __init__.py
│
├── main.py                          # Entry point for training and evaluation
├── test_enhancements.py            # Unit tests for augmentation, projection, clustering
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Model Architecture

### Overall Pipeline

```
Input Time Series (N, C, L)
    ↓
CNN Feature Extractor
    ↓
Feature Vector (N, 256)
    ↓
Prototype Module (with optional projection)
    ↓
Attended Features (N, 256)
    ↓
Linear Classifier
    ↓
Class Logits (N, num_classes)
```

### 1. CNN Feature Extractor

Three-layer 1D convolutional network with batch normalization and dropout:

- **Layer 1:** Conv1d(C → 64, kernel=7, padding=3) + BatchNorm + ReLU + Dropout + MaxPool(2)
- **Layer 2:** Conv1d(64 → 128, kernel=5, padding=2) + BatchNorm + ReLU + Dropout + MaxPool(2)
- **Layer 3:** Conv1d(128 → 256, kernel=3, padding=1) + BatchNorm + ReLU + Dropout
- **Pooling:** Adaptive average pooling to produce fixed 256-dimensional feature vector

Input shape: `(batch_size, channels, length)` or `(batch_size, length)` for univariate  
Output shape: `(batch_size, 256)`

### 2. Prototype Module

Learns `num_prototypes` learnable prototype vectors in feature space (or optional projection space).

**Without Projection:**
- Prototypes: `(num_prototypes, 256)`
- Similarity: Cosine similarity between L2-normalized features and prototypes
- Attention: `softmax(similarity / temperature)`
- Output: `features + attention @ prototypes` (residual connection)

**With Projection (optional):**
- Projects features to lower-dimensional space via linear layer
- Computes attention in projection space
- Projects attended features back to original 256-dimensional space
- Enables learning in a more compact representation

**Initialization:**
- K-Means clustering on extracted features from training set
- Fallback to random initialization if K-Means fails (insufficient samples)

**Diversity Loss:**
- Encourages prototype separation by penalizing pairwise cosine similarity
- Loss: `mean(|prototype_i · prototype_j|)` for i ≠ j

### 3. Clustering Loss (Optional)

Two-component loss to improve prototype quality:

**Compactness Loss:**
- Minimizes weighted distance between features and assigned prototypes
- Distance metric: `1 - cosine_similarity`
- Weighted by attention scores (soft assignment)

**Separation Loss:**
- Maximizes pairwise distances between different prototypes
- Computed as negative average cosine similarity between prototypes

Total clustering loss: `compactness_weight × L_compact - separation_weight × L_separate`

### 4. Training Objective

```
Total Loss = CrossEntropy(logits, labels)
           + diversity_weight × DiversityLoss(prototypes)
           + clustering_weight × ClusteringLoss(features, prototypes, attention)
```

Hyperparameters:
- `diversity_weight`: Default 0.01
- `clustering_weight`: Default 0.1
- `compactness_weight`: Default 1.0
- `separation_weight`: Default 0.5

### 5. Dynamic Prototype Allocation

Number of prototypes determined by dataset complexity:
- `num_classes ≤ 20`: `num_prototypes = num_classes × 2` (lightweight)
- `num_classes > 20`: `num_prototypes = num_classes × 5` (high capacity)

User can override via `config.yaml` by setting `num_prototypes` to a specific integer.

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (optional, CPU supported)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - K-Means initialization and metrics
- `matplotlib>=3.7.0` - Visualization
- `tslearn>=0.6.0` - Time series utilities
- `PyYAML>=6.0` - Configuration parsing
- `scipy>=1.10.0` - Scientific computing
- `aeon>=0.7.0` - Time series datasets
- `pandas>=2.0.0` - Data manipulation

## Usage

### Dataset Preparation

Place datasets in `datasets/` directory with the following structure:

```
datasets/
└── DatasetName/
    ├── DatasetName_TRAIN.tsv  (or .ts, .txt, .csv, .arff)
    └── DatasetName_TEST.tsv
```

Supported formats:
- `.tsv` - Tab-separated values (label in first column)
- `.ts` - Time series format with colon-separated dimensions
- `.txt` - Space-separated values
- `.csv` - Comma-separated values
- `.arff` - Weka ARFF format

### Training

**Single dataset:**
```bash
python main.py --dataset GunPoint
```

**All available datasets:**
```bash
python main.py --all
```

**List datasets:**
```bash
python main.py --list
```

**Dataset information:**
```bash
python main.py --dataset ECG200 --info
```

**Disable result saving:**
```bash
python main.py --dataset GunPoint --no-save
```

### Configuration

Edit `config.yaml` to modify hyperparameters:

```yaml
default:
  # Model architecture
  num_prototypes: null          # null = auto (2x or 5x based on num_classes)
  dropout: 0.1
  temperature: 0.5
  use_projection: true
  projection_dim: null          # null = same as feature_dim
  
  # Training
  batch_size: 8
  epochs: 500
  learning_rate: 0.0005
  weight_decay: 0.0001
  early_stopping_patience: 50
  
  # Loss weights
  diversity_weight: 0.01
  clustering_weight: 0.1
  label_smoothing: 0.1
  
  # Augmentation
  use_augmentation: true
  augmentation:
    jitter_std: 0.03
    scaling_range: [0.8, 1.2]
    time_warp_strength: 0.2
    window_slice_ratio: 0.9
    rotation_prob: 0.5
    augment_prob: 0.5
  
  # Visualization
  plot_training: true
  plot_pca: true
```

### Output Structure

Results are saved per dataset in `results/DatasetName/`:

```
results/
└── DatasetName/
    ├── current.json                    # Latest run results
    ├── best.json                       # Best run results (highest accuracy)
    ├── history.json                    # All run history
    └── visualizations/
        ├── current/                    # Latest run visualizations
        │   ├── confusion_matrix.png
        │   ├── pca_visualization.png
        │   ├── prototype_heatmap.png
        │   ├── sample_predictions.png
        │   └── training_curves.png
        └── best/                       # Best run visualizations
            └── (same as current)
```

JSON structure:
```json
{
  "run_info": {
    "dataset": "GunPoint",
    "timestamp": "2025-11-25T12:00:00",
    "run_id": "20251125_120000"
  },
  "performance": {
    "test_accuracy": 0.9867,
    "best_epoch": 45,
    "total_epochs": 95,
    "training_time_seconds": 123.45
  },
  "dataset_info": {
    "type": "UNIVARIATE",
    "channels": 1,
    "classes": 2,
    "time_steps": 150
  },
  "hyperparameters": { ... },
  "training_summary": {
    "loss": {"min": 0.0234, "final": 0.0456},
    "train_accuracy": {"best": 0.9912, "final": 0.9876},
    "test_accuracy": {"best": 0.9867, "final": 0.9867},
    "generalization_gap": {"at_best_epoch": 0.0045, "final": 0.0009}
  }
}
```

### Adaptive Batch Size

For small datasets, batch size is automatically adjusted:
```
adaptive_batch_size = max(8, min(config_batch_size, num_samples // 10))
```

Ensures minimum 8 samples per batch for stable gradients while respecting dataset size.

### Data Normalization

Instance normalization is applied per sample:
```
normalized = (x - mean(x, axis=time)) / (std(x, axis=time) + 1e-8)
```

Computed independently for each channel in multivariate series.

## Testing

Run unit tests for enhancements:
```bash
python test_enhancements.py
```

Tests cover:
- Data augmentation (univariate, multivariate, batch)
- Projection layer (with/without, different dimensions)
- Clustering loss (compactness, separation, gradient flow)
