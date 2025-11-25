# Results Directory

This directory contains all experimental results, visualizations, and performance metrics for the CNNProto model.

## Directory Structure

```
results/
├── {DatasetName}/
│   ├── current.json                    # Latest experiment results
│   ├── best.json                       # Best performing run
│   ├── history.json                    # All experiment runs
│   ├── visualizations/
│   │   ├── current/                    # Latest run visualizations
│   │   │   ├── pca_visualization.png
│   │   │   ├── training_curves.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── prototype_heatmap.png
│   │   │   └── sample_predictions.png
│   │   └── best/                       # Best run visualizations
│   │       └── (same files as current)
│   └── evaluation/
│       └── results.json                # Evaluation script results
├── comparison_table.csv                # Cross-dataset comparison
├── comparison_table.xlsx               # Excel version
└── summary_YYYYMMDD_HHMMSS.json       # Batch run summary
```

## File Formats

### Result JSON Files

**`current.json`** - Latest experiment:
```json
{
  "run_id": "20251125_123456",
  "timestamp": "2025-11-25T12:34:56",
  "dataset_name": "GunPoint",
  "hyperparameters": {
    "num_prototypes": 4,
    "dropout": 0.1,
    "temperature": 0.5,
    "batch_size": 8,
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "diversity_weight": 0.05,
    "use_augmentation": true,
    "augmentation_config": {...},
    "use_projection": true,
    "projection_dim": 256,
    "clustering_weight": 0.1,
    "clustering_config": {...}
  },
  "dataset_info": {
    "input_channels": 1,
    "num_classes": 2,
    "time_steps": 150,
    "train_size": 50,
    "test_size": 150
  },
  "final_metrics": {
    "test_accuracy": 0.9867,
    "best_epoch": 127,
    "total_epochs": 177,
    "training_time": 45.32,
    "total_parameters": 274178
  },
  "training_summary": {
    "best_train_acc": 1.0,
    "best_test_acc": 0.9867,
    "final_train_acc": 1.0,
    "final_test_acc": 0.98,
    "min_train_loss": 0.0234,
    "min_test_loss": 0.0456,
    "generalization_gap": 0.0133
  }
}
```

**`best.json`** - Best performing run (same format as `current.json`)

**`history.json`** - All experiment runs:
```json
[
  {
    "run_id": "20251125_120000",
    "timestamp": "2025-11-25T12:00:00",
    "accuracy": 0.9733,
    "epochs": 150,
    "training_time": 42.1,
    "is_best": false
  },
  {
    "run_id": "20251125_123456",
    "timestamp": "2025-11-25T12:34:56",
    "accuracy": 0.9867,
    "epochs": 177,
    "training_time": 45.32,
    "is_best": true
  }
]
```

### Summary Files

**`summary_YYYYMMDD_HHMMSS.json`** - Batch run summary:
```json
{
  "timestamp": "2025-11-25T15:30:00",
  "total_datasets": 12,
  "results": [
    {
      "dataset": "GunPoint",
      "accuracy": 0.9867,
      "epochs": 177,
      "time": 45.32,
      "parameters": 274178
    },
    ...
  ],
  "statistics": {
    "mean_accuracy": 0.8542,
    "std_accuracy": 0.0823,
    "total_time": 542.1
  }
}
```

### Comparison Tables

**`comparison_table.csv`** - Cross-dataset comparison:
```csv
Dataset,Train Size,Test Size,Classes,Accuracy,Epochs,Time (s),Parameters
GunPoint,50,150,2,0.9867,177,45.32,274178
Coffee,28,28,2,0.9643,234,32.15,274178
...
```

## Visualizations

### 1. PCA Visualization (`pca_visualization.png`)
- 2D PCA projection of learned features
- Prototypes shown as stars
- Color-coded by class
- Shows feature space organization

### 2. Training Curves (`training_curves.png`)
- **Top**: Training and test loss over epochs
- **Middle**: Training and test accuracy over epochs
- **Bottom**: Generalization gap (train_acc - test_acc)
- Vertical line indicates best epoch

### 3. Confusion Matrix (`confusion_matrix.png`)
- **Left**: Count-based confusion matrix
- **Right**: Normalized confusion matrix
- Shows per-class performance
- Diagonal = correct predictions

### 4. Prototype Heatmap (`prototype_heatmap.png`)
- Learned prototype patterns
- Each row = one prototype
- Shows temporal patterns captured
- Useful for interpretability

### 5. Sample Predictions (`sample_predictions.png`)
- 6 random test samples
- Original time series plot
- Predicted vs true label
- Confidence score
- Visual verification of model performance

## Usage

### Accessing Results

```python
from utils.results_manager import ResultsManager

# Initialize manager
results_manager = ResultsManager()

# Load best result for a dataset
best_result = results_manager.load_best_result("GunPoint")
print(f"Best accuracy: {best_result['final_metrics']['test_accuracy']}")

# Load history
history = results_manager.load_history("GunPoint")
print(f"Total runs: {len(history)}")
```

### Generating Comparison Tables

```python
# After running multiple datasets
python main.py --all

# Results automatically saved to:
# - results/summary_YYYYMMDD_HHMMSS.json
# - results/comparison_table.csv
# - results/comparison_table.xlsx
```

## Best Practices

### 1. Result Tracking
- **current.json**: Always updated with latest run
- **best.json**: Only updated when accuracy improves
- **history.json**: Keeps all runs for analysis

### 2. Visualization Management
- **current/**: Latest experiment visualizations
- **best/**: Automatically copied when new best is achieved
- Compare current vs best to see improvements

### 3. Experiment Comparison
- Use `history.json` to track hyperparameter tuning
- Use `comparison_table.csv` for cross-dataset analysis
- Use `summary_*.json` for batch run statistics

## Metrics Explained

### Accuracy
- Primary metric: Test set accuracy
- Range: [0.0, 1.0]
- Higher is better

### Generalization Gap
- `gap = train_accuracy - test_accuracy`
- Lower gap = better generalization
- High gap may indicate overfitting

### Training Time
- Wall-clock time in seconds
- Includes data loading, training, and evaluation
- Useful for efficiency comparison

### Parameters
- Total number of model parameters
- Varies with:
  - `use_projection`: +131K params
  - `num_prototypes`: ~1K params per prototype

## Notes

### Result Persistence
- Results are never overwritten (except `current.json`)
- Each run gets unique timestamp
- History accumulates all experiments

### Storage Considerations
- Visualizations: ~5MB per dataset
- JSON files: ~100KB per dataset
- Consider cleaning old runs periodically

### Reproducibility
- All hyperparameters saved in result files
- Random seed tracked
- Dataset info included
- Can reproduce any experiment from saved config

## Troubleshooting

### Missing Visualizations
- Check if `plot_pca` and `plot_training` are enabled in config
- Ensure matplotlib backend is properly configured
- Check file permissions in results directory

### Large History Files
- History grows with each run
- Consider archiving old runs
- Can manually edit `history.json` if needed

### Comparison Table Not Generated
- Only created when using `--all` flag
- Requires multiple dataset results
- Check write permissions

---

For more information, see the main [README.md](../README.md) in the project root.
