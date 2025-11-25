import os
from pathlib import Path
from typing import List, Tuple, Dict


def discover_datasets(datasets_dir: str = "datasets") -> List[str]:
    """Discover all available datasets in the datasets directory.

    Args:
        datasets_dir: Directory containing datasets.

    Returns:
        List of dataset names that have both TRAIN and TEST files.
    """
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        return []

    valid_datasets = []

    for item in datasets_path.iterdir():
        if item.is_dir():
            dataset_name = item.name
            if validate_dataset(dataset_name, datasets_dir):
                valid_datasets.append(dataset_name)

    return sorted(valid_datasets)


def validate_dataset(dataset_name: str, datasets_dir: str = "datasets") -> bool:
    """Validate that a dataset has required TRAIN and TEST files.

    Args:
        dataset_name: Name of the dataset.
        datasets_dir: Directory containing datasets.

    Returns:
        True if dataset is valid, False otherwise.
    """
    dataset_path = Path(datasets_dir) / dataset_name
    
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False

    train_file = dataset_path / f"{dataset_name}_TRAIN.tsv"
    test_file = dataset_path / f"{dataset_name}_TEST.tsv"

    return train_file.exists() and test_file.exists()


def list_datasets(datasets_dir: str = "datasets", verbose: bool = True) -> List[Tuple[str, bool]]:
    """List all datasets with their validation status.

    Args:
        datasets_dir: Directory containing datasets.
        verbose: Whether to print the list to console.

    Returns:
        List of tuples (dataset_name, is_valid).
    """
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        if verbose:
            print(f"Datasets directory not found: {datasets_dir}")
        return []

    all_items = [(item.name, validate_dataset(item.name, datasets_dir)) 
                 for item in datasets_path.iterdir() if item.is_dir()]
    
    all_items.sort(key=lambda x: x[0])

    if verbose:
        print("\nAvailable Datasets")
        print("-" * 70)
        
        if not all_items:
            print("No datasets found.")
            print(f"\nPlease add datasets to: {datasets_path.absolute()}")
            print("Each dataset should have:")
            print("  - {DatasetName}_TRAIN.tsv")
            print("  - {DatasetName}_TEST.tsv")
        else:
            valid_count = sum(1 for _, valid in all_items if valid)
            print(f"Found {len(all_items)} dataset(s), {valid_count} valid\n")
            
            for name, is_valid in all_items:
                status = "VALID" if is_valid else "INVALID"
                print(f"  {status:<8} {name}")
        
        print("-" * 70)

    return all_items
