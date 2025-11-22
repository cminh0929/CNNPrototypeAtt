# Quickstart Guide

## Prerequisites

Ensure you have Python 3.8 or higher installed.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/cminh0929/CNNPrototypeAtt.git
cd CNNPrototypeAtt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Download Dataset

1. Visit the UCR Time Series Archive:
   - https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

2. Download a dataset (e.g., GunPoint)

3. Extract and place in `datasets/`:
```
datasets/
└── GunPoint/
    ├── GunPoint_TRAIN.tsv
    └── GunPoint_TEST.tsv
```

## Run Training

1. Open `main.py` and set the dataset name:
```python
def main(dataset_name: str = "GunPoint") -> None:
```

2. Run the training script:
```bash
python main.py
```

## Expected Output

The script will:
1. Load and preprocess the dataset
2. Train the model with early stopping
3. Evaluate on the test set
4. Generate visualizations in the current directory

## Configuration

To modify training parameters, edit `config.yaml`:

```yaml
default:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

For dataset-specific settings, add a section with the dataset name:

```yaml
GunPoint:
  batch_size: 16
  epochs: 150
```

## Common Datasets

Small datasets for quick testing:
- GunPoint (50 train, 150 test)
- Coffee (28 train, 28 test)
- ECG200 (100 train, 100 test)

Medium datasets:
- FordA (3601 train, 1320 test)
- Wafer (1000 train, 6164 test)

## Troubleshooting

**CUDA out of memory**: Reduce `batch_size` in `config.yaml`

**Dataset not found**: Verify the dataset folder structure matches the required format

**Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
