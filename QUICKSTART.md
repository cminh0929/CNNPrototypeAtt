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

### List Available Datasets

```bash
python main.py --list
```

### Run on Single Dataset

```bash
# Run on default dataset (GunPoint)
python main.py

# Run on specific dataset
python main.py --dataset ECG200
```

### Run on All Datasets

```bash
python main.py --all
```

## Command-Line Options

- `--dataset NAME`: Run experiment on specific dataset (default: GunPoint)
- `--all`: Run experiments on all available datasets
- `--list`: List all available datasets and exit
- `--no-save`: Don't save results to disk

## Expected Output

The script will:
1. Load and preprocess the dataset
2. Train the model with early stopping
3. Evaluate on the test set
4. Generate visualizations in the current directory
5. Save results to `results/{dataset_name}/` directory

## Results

Results are automatically saved in JSON format:
```
results/
├── GunPoint/
│   ├── GunPoint_20251124_163525.json
│   └── latest.json
└── summary_20251124_171500.json (when using --all)
```

Each result file contains:
- Configuration used
- Training history (loss, accuracy per epoch)
- Final metrics (accuracy, training time, etc.)
- Dataset information

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
