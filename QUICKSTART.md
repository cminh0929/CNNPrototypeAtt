# 🚀 Quickstart Guide

Get started with CNNProto in 5 minutes!

## ⚡ Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### 2. Prepare Datasets

CNNProto supports **automatic dataset loading** via `aeon` for UCR Archive datasets:

```bash
# List all available datasets
python main.py --list

# Run on a dataset (auto-downloads if needed)
python main.py --dataset GunPoint
```

**Manual Dataset Setup** (optional):

If you prefer local files, download from [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and place in `datasets/`:

```
datasets/
└── GunPoint/
    ├── GunPoint_TRAIN.tsv
    └── GunPoint_TEST.tsv
```

## 🎯 Basic Usage

### Run Single Experiment

```bash
# Default dataset (GunPoint)
python main.py

# Specific dataset
python main.py --dataset ECG200

# Custom seed for reproducibility
python main.py --dataset Coffee --seed 42
```

### Run Benchmarks

```bash
# All datasets sequentially
python main.py --all

# Windows batch script with multiple seeds
run_benchmark.bat
```

## 📊 Results & Outputs

### Directory Structure

```
results/
├── GunPoint/
│   ├── current.json          # Latest run results
│   ├── best.json             # Best performance achieved
│   ├── history.json          # All previous runs
│   └── visualizations/
│       ├── current/          # Latest visualizations
│       └── best/             # Best run visualizations
└── summary_YYYYMMDD_HHMMSS.json  # Benchmark summary
```

### Visualizations Generated

- **Training curves**: Loss and accuracy over epochs
- **Confusion matrix**: Classification performance breakdown
- **Prototype analysis**: Learned prototype patterns
- **PCA visualization**: Feature space representation
- **Sample predictions**: Model predictions on test samples

## ⚙️ Configuration

### Basic Configuration

Edit `config/config.yaml` for global settings:

```yaml
default:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  num_prototypes: 10
  patience: 15
```

### Dataset-Specific Overrides

Create `config/{dataset_name}.yaml` for custom settings:

```yaml
# config/GunPoint.yaml
batch_size: 16
epochs: 150
learning_rate: 0.0005
```

## 🎛️ Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset NAME` | Specify dataset to run | `GunPoint` |
| `--all` | Run on all available datasets | `False` |
| `--list` | List available datasets and exit | - |
| `--seed N` | Set random seed | `42` |
| `--no-save` | Skip saving results | `False` |
| `--config PATH` | Custom config file | `config/config.yaml` |

## 📈 Recommended Datasets

### Quick Testing (< 1 min)
- **GunPoint**: 50 train, 150 test, 2 classes
- **Coffee**: 28 train, 28 test, 2 classes
- **ItalyPowerDemand**: 67 train, 1029 test, 2 classes

### Standard Benchmarks (1-5 min)
- **ECG200**: 100 train, 100 test, 2 classes
- **FaceFour**: 24 train, 88 test, 4 classes
- **Beef**: 30 train, 30 test, 5 classes

### Larger Datasets (5-30 min)
- **FordA**: 3601 train, 1320 test, 2 classes
- **Wafer**: 1000 train, 6164 test, 2 classes
- **ElectricDevices**: 8926 train, 7711 test, 7 classes

## 🔧 Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config.yaml
batch_size: 16  # or 8
```

### Dataset Not Found
```bash
# Verify dataset name
python main.py --list

# Check datasets/ folder structure
# Ensure TRAIN/TEST files match dataset name
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

### Slow Training
- Enable CUDA: Ensure PyTorch detects GPU
- Reduce dataset size: Use smaller datasets for testing
- Adjust `num_prototypes`: Lower values train faster

## 📚 Next Steps

- **Full Documentation**: See [README.md](README.md) for architecture details
- **Custom Datasets**: Check `data/dataloader_manager.py` for format requirements
- **Model Tuning**: Explore `config/` for advanced hyperparameters
- **Benchmarking**: Use `run_benchmark.bat` for systematic evaluation

## 💡 Tips

1. **Start small**: Test with `GunPoint` or `Coffee` first
2. **Monitor GPU**: Use `nvidia-smi` to check GPU utilization
3. **Save best models**: Results are auto-tracked in `best.json`
4. **Compare runs**: Check `history.json` for performance trends
5. **Visualize results**: All plots saved in `visualizations/` folders

---
