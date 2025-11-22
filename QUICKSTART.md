# Quick Start Guide

## ✅ Virtual Environment Setup Complete!

Your Python 3.14 virtual environment is ready with all dependencies installed.

## 🚀 How to Use

### 1. Activate the Virtual Environment

**Option A: Use the activation script (Recommended)**
```bash
activate.bat
```

**Option B: Manual activation**
```bash
venv\Scripts\activate
```

You'll see `(venv)` in your command prompt when activated.

### 2. Run the Code

```bash
python main.py
```

### 3. Deactivate (when done)

```bash
deactivate
```

## 📦 Installed Packages

- ✅ PyTorch 2.9.1
- ✅ NumPy 2.3.5
- ✅ scikit-learn 1.7.2
- ✅ matplotlib 3.10.7
- ✅ tslearn 0.7.0
- ✅ PyYAML 6.0.3

## 🎯 Quick Test

After activating the environment, test if everything works:

```bash
python -c "import torch; import numpy; import sklearn; import matplotlib; import tslearn; print('All imports successful!')"
```

## 📝 Notes

- **Python Version**: 3.14.0 (newer than requested 3.10, but fully compatible)
- **tslearn**: Installed without numba dependency (numba doesn't support Python 3.14 yet)
- **GPU Support**: PyTorch will automatically use CUDA if available

## 🔄 Changing Datasets

Edit `main.py` line 15 to change the dataset:

```python
def main(dataset_name: str = "GunPoint") -> None:  # Change dataset here
```

Popular options:
- `GunPoint` - Small, fast
- `ECG200` - ECG signals
- `Coffee` - Coffee classification
- `ElectricDevices` - Default dataset

## 🐛 Troubleshooting

### Virtual environment not activating?
Make sure you're in the project directory:
```bash
cd c:\Users\cminh\Desktop\Code\CNNProto
```

### Import errors?
Verify packages are installed:
```bash
venv\Scripts\python.exe -m pip list
```

### Need to reinstall?
```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```
