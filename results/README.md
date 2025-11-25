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