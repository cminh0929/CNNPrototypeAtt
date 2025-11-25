# Results Structure Documentation

## Overview

The results management system now tracks best models, maintains run history, and organizes visualizations intelligently.

## Directory Structure

```
results/
└── GunPoint/
    ├── current.json              # Latest run results
    ├── best.json                 # Best run results (highest accuracy)
    ├── history.json              # All runs history
    └── visualizations/
        ├── current/              # Latest run visualizations
        │   ├── pca_visualization.png
        │   ├── training_curves.png
        │   ├── confusion_matrix.png
        │   ├── prototype_heatmap.png
        │   └── sample_predictions.png
        └── best/                 # Best run visualizations (auto-updated)
            ├── pca_visualization.png
            ├── training_curves.png
            ├── confusion_matrix.png
            ├── prototype_heatmap.png
            └── sample_predictions.png
```

## File Descriptions

### current.json
Contains results from the most recent run with simplified, readable format:

```json
{
  "run_info": {
    "dataset": "GunPoint",
    "timestamp": "2025-11-24T17:12:00",
    "run_id": "20251124_171200"
  },
  "performance": {
    "test_accuracy": 0.9267,
    "best_epoch": 87,
    "total_epochs": 107,
    "training_time_seconds": 45.23,
    "model_parameters": 143106
  },
  "dataset_info": {
    "type": "UNIVARIATE",
    "channels": 1,
    "classes": 2,
    "time_steps": 150,
    "train_samples": 50,
    "test_samples": 150
  },
  "hyperparameters": {
    "num_prototypes": 6,
    "dropout": 0.2,
    "temperature": 0.5,
    "batch_size": 16,
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "diversity_weight": 0.01
  },
  "training_summary": {
    "loss": {
      "min": 0.3421,
      "final": 0.4156
    },
    "train_accuracy": {
      "best": 0.9200,
      "final": 0.8800
    },
    "test_accuracy": {
      "best": 0.9267,
      "final": 0.9267
    },
    "generalization_gap": {
      "at_best_epoch": 0.0067,
      "final": -0.0467
    }
  }
}
```

### best.json
Same format as `current.json`, but contains the run with highest test accuracy.
**Only updated when a new run achieves better accuracy.**

### history.json
Tracks all runs with summary information:

```json
[
  {
    "timestamp": "2025-11-24T16:35:25",
    "run_id": "20251124_163525",
    "accuracy": 0.8533,
    "epochs": 95,
    "training_time": 38.5,
    "is_best": false
  },
  {
    "timestamp": "2025-11-24T17:01:48",
    "run_id": "20251124_170148",
    "accuracy": 0.7467,
    "epochs": 15,
    "training_time": 3.06,
    "is_best": false
  },
  {
    "timestamp": "2025-11-24T17:12:00",
    "run_id": "20251124_171200",
    "accuracy": 0.9267,
    "epochs": 107,
    "training_time": 45.23,
    "is_best": true
  }
]
```

## Behavior

### When Running an Experiment

1. **Always saves to `current.json`** - Latest run results
2. **Always saves visualizations to `visualizations/current/`** - Latest run plots
3. **Compares with `best.json`**:
   - If current accuracy > best accuracy (or no best exists):
     - Updates `best.json`
     - Copies visualizations to `visualizations/best/`
     - Prints "NEW BEST RESULT!" message
   - Otherwise:
     - Keeps existing `best.json` unchanged
     - Keeps existing `visualizations/best/` unchanged
     - Prints current vs best comparison
4. **Appends to `history.json`** - Adds run summary with timestamp

### Advantages

✅ **Easy comparison**: Compare current vs best results side-by-side
✅ **Track progress**: See all runs in history.json
✅ **Best model preserved**: Best visualizations never overwritten unless improved
✅ **Readable JSON**: Simplified structure, clear field names
✅ **Organized**: Separate folders for current and best visualizations

## Usage Examples

### View Current Results
```python
from utils.results_manager import ResultsManager

rm = ResultsManager()
current = rm.load_result("GunPoint", "current.json")
print(f"Current accuracy: {current['performance']['test_accuracy']}")
```

### View Best Results
```python
best = rm.load_result("GunPoint", "best.json")
print(f"Best accuracy: {best['performance']['test_accuracy']}")
```

### View Run History
```python
rm.print_history("GunPoint", limit=10)
```

Output:
```
Run History for GunPoint
--------------------------------------------------------------------------------
Run ID             Accuracy     Epochs   Time (s)   Best
--------------------------------------------------------------------------------
20251124_163525    0.8533 (85.33%) 95       38.50      
20251124_170148    0.7467 (74.67%) 15       3.06       
20251124_171200    0.9267 (92.67%) 107      45.23      YES
--------------------------------------------------------------------------------
Total runs: 3
```
