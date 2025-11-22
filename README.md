# CNN Prototype Attention Model for Time Series Classification

## Overview

This repository implements a CNN-based prototype attention model for time series classification. The model learns interpretable prototypes in the feature space and uses attention mechanisms to make predictions based on similarity to these prototypes.

## Requirements

- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- NumPy 1.24.0 or higher
- scikit-learn 1.3.0 or higher
- matplotlib 3.7.0 or higher
- PyYAML 6.0 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cminh0929/CNNPrototypeAtt.git
cd CNNPrototypeAtt
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

This project uses datasets from the UCR Time Series Archive.

1. Download datasets from:
   - https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
   - http://timeseriesclassification.com/

2. Extract the downloaded files and place them in the `datasets/` directory with the following structure:
```
datasets/
├── DatasetName1/
│   ├── DatasetName1_TRAIN.tsv
│   └── DatasetName1_TEST.tsv
├── DatasetName2/
│   ├── DatasetName2_TRAIN.tsv
│   └── DatasetName2_TEST.tsv
└── ...
```

Each dataset folder must contain two files:
- `{DatasetName}_TRAIN.tsv` - Training data
- `{DatasetName}_TEST.tsv` - Test data

## Project Structure

```
CNNProto/
├── config/
│   └── config_manager.py      # Configuration management
├── data/
│   ├── dataloader_manager.py  # Data loading and preprocessing
│   └── dataset.py             # PyTorch Dataset wrapper
├── models/
│   ├── cnn_backbone.py        # CNN feature extractor
│   ├── cnn_proto_attention.py # Complete model architecture
│   └── prototype.py           # Prototype learning module
├── training/
│   ├── evaluator.py           # Model evaluation
│   └── trainer.py             # Training loop
├── utils/
│   ├── device.py              # Device management
│   └── seed.py                # Random seed setting
├── visualization/
│   └── visualizer.py          # Visualization utilities
├── config.yaml                # Configuration file
├── main.py                    # Main entry point
└── requirements.txt           # Python dependencies
```

## Configuration

Model and training parameters can be configured in `config.yaml`. The configuration file supports:

- Default parameters applied to all datasets
- Dataset-specific parameter overrides

Example configuration:
```yaml
default:
  num_prototypes: null        # Auto-calculate as num_classes * 2
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
  plot_pca: true
  plot_training: true

DatasetName:
  num_prototypes: 6
  batch_size: 16
  epochs: 150
```

## Training

To train the model on a specific dataset, modify the `dataset_name` parameter in `main.py`:

```python
def main(dataset_name: str = "GunPoint") -> None:
    ...
```

Then run:
```bash
python main.py
```

The training process will:
1. Load and preprocess the dataset
2. Initialize the model with specified parameters
3. Train with early stopping based on validation accuracy
4. Evaluate on the test set
5. Generate visualizations

## Evaluation

The model is evaluated using:
- Accuracy
- F1-score (macro-averaged)
- Per-class precision, recall, and F1-score

Evaluation results are printed to the console and include a detailed classification report.

## Visualization

The following visualizations are automatically generated during training:

1. `pca_visualization.png` - PCA projection of learned features and prototypes
2. `training_curves.png` - Training loss, accuracy, and generalization gap
3. `confusion_matrix.png` - Confusion matrix (counts and normalized)
4. `prototype_heatmap.png` - Learned prototype patterns
5. `sample_predictions.png` - Sample predictions with confidence scores

## Model Architecture

The model consists of three main components:

1. **CNN Backbone**: Extracts features from input time series
2. **Prototype Layer**: Learns interpretable prototypes in feature space
3. **Attention Mechanism**: Computes similarity-based attention weights for classification

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cnn_prototype_attention,
  author = {Your Name},
  title = {CNN Prototype Attention Model for Time Series Classification},
  year = {2025},
  url = {https://github.com/cminh0929/CNNPrototypeAtt}
}
```

## License

This project is licensed under the MIT License.
