# CNNProto - Time Series Classification with Prototypes

A CNN-based prototype attention model for time series classification using UCR/UEA datasets.

## 🚀 Quick Start

### 1. Install Dependencies

First, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- scikit-learn (metrics and PCA)
- matplotlib (visualization)
- tslearn (UCR/UEA time series datasets)
- PyYAML (configuration files)

### 2. Run the Code

#### Basic Usage

Run with the default dataset (ElectricDevices):

```bash
python main.py
```

#### Run with Different Datasets

You can modify the dataset in `main.py` by changing the default parameter, or run it programmatically:

```python
from main import main

# Run with a specific dataset
main("GunPoint")
main("ECG200")
main("FordA")
main("Coffee")
```

### 3. Available UCR/UEA Datasets

Popular datasets you can try:
- `GunPoint` - Small dataset, good for testing
- `ECG200` - ECG signals classification
- `FordA` - Engine noise classification
- `Wafer` - Semiconductor wafer classification
- `Coffee` - Coffee/non-coffee classification
- `ElectricDevices` - Electric device classification (default)
- And many more from the UCR archive!

## 📁 Project Structure

```
CNNProto/
├── config/
│   ├── __init__.py
│   └── config_manager.py      # Configuration management
├── data/
│   ├── __init__.py
│   ├── dataloader_manager.py  # Data loading and preprocessing
│   └── dataset.py             # PyTorch Dataset wrapper
├── models/
│   ├── __init__.py
│   ├── cnn_backbone.py        # CNN feature extractor
│   ├── cnn_proto_attention.py # Complete model
│   └── prototype.py           # Prototype learning module
├── training/
│   ├── __init__.py
│   ├── evaluator.py           # Model evaluation
│   └── trainer.py             # Training loop
├── utils/
│   ├── __init__.py
│   ├── device.py              # Device management
│   └── seed.py                # Random seed setting
├── visualization/
│   ├── __init__.py
│   └── visualizer.py          # PCA and training plots
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
└── main.py                    # Main entry point
```

## ⚙️ Configuration

Edit `config.yaml` to customize hyperparameters:

### Default Settings
```yaml
default:
  num_prototypes: null        # Auto-calculate (num_classes * 2)
  dropout: 0.1
  temperature: 1.0
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  label_smoothing: 0.0
  diversity_weight: 0.01
  early_stopping_patience: 20
  seed: 42
  device: auto                # 'auto', 'cuda', or 'cpu'
```

### Dataset-Specific Settings

You can override default settings for specific datasets:

```yaml
GunPoint:
  num_prototypes: 6
  dropout: 0.2
  batch_size: 16
  epochs: 150
```

## 🎯 What the Code Does

1. **Loads Configuration**: Reads settings from `config.yaml`
2. **Sets Random Seed**: Ensures reproducibility
3. **Loads Dataset**: Downloads and preprocesses UCR/UEA dataset
4. **Creates Model**: Builds CNN + Prototype Attention model
5. **Trains Model**: Trains with early stopping and learning rate scheduling
6. **Evaluates**: Tests on test set and shows metrics
7. **Visualizes**: 
   - PCA plot of features and prototypes
   - Training curves (loss and accuracy)

## 📊 Output

The code will:
- Print training progress (loss, accuracy per epoch)
- Show final evaluation metrics (accuracy, F1-score, classification report)
- Generate visualizations:
  - `pca_visualization.png` - PCA plot of learned features
  - `training_curves.png` - Training history plots

## 💡 Example Output

```
======================================================================
DATA LOADING: GunPoint
======================================================================
Raw shape: (50, 150)
 UNIVARIATE
 Applying PER-CHANNEL Z-SCORE NORMALIZATION... Done (univariate global norm)
Classes: 2
Time steps: 150
Train size: 50 | Test size: 150
======================================================================

 Model: 234,562 parameters

======================================================================
TRAINING
======================================================================
Epoch   1/150 | Loss: 0.6234 | Train: 0.7200 | Test: 0.8133 | LR: 0.001000
Epoch   2/150 | Loss: 0.4521 | Train: 0.8400 | Test: 0.8667 | LR: 0.001000
...
======================================================================

======================================================================
EVALUATION RESULTS
======================================================================
Accuracy: 0.9133 (91.33%)
F1-Score: 0.9128
...
======================================================================

======================================================================
FINAL SUMMARY
======================================================================
Dataset:        GunPoint
Type:           UNIVARIATE
Channels:       1
Classes:        2
Final accuracy: 0.9133 (91.33%)
======================================================================
```

## 🔧 Troubleshooting

### CUDA Out of Memory
Reduce batch size in `config.yaml`:
```yaml
batch_size: 16  # or 8
```

### Dataset Download Issues
The first run will download the dataset. Ensure you have internet connection.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📝 Notes

- First run will download the dataset (may take a few minutes)
- GPU is automatically used if available
- All code uses type hints and PEP 257 docstrings
- Training uses early stopping to prevent overfitting
