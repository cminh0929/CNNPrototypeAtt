# Datasets

Datasets are not stored in this repository.

## Download Instructions

1. Download datasets from the UCR Time Series Archive:
   - Visit: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
   - Or use: http://timeseriesclassification.com/

2. Extract the downloaded dataset files.

3. Place the dataset folders in this `datasets/` directory.

## Expected Structure

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

Each dataset folder must contain:
- `{DatasetName}_TRAIN.tsv` - Training data
- `{DatasetName}_TEST.tsv` - Test data
