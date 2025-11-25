import yaml
import json
from typing import Dict, Any
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and retrieval for datasets."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the ConfigManager.

        Args:
            config_path: Path to the configuration file.
        """
        self.config_path: Path = Path(config_path)
        self.configs: Dict[str, Any] = self._load_config()
        self.default_config: Dict[str, Any] = self.configs.get('default', self._get_default_config())
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration dictionary."""
        return {
            'num_prototypes': None,
            'dropout': 0.1,
            'temperature': 1.0,
            'batch_size': 8,
            'epochs': 200,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'label_smoothing': 0.0,
            'diversity_weight': 0.05,
            'grad_clip': 1.0,
            'early_stopping_patience': 100,
            'lr_scheduler': 'plateau',
            'lr_patience': 10,
            'lr_factor': 0.5,
            'save_checkpoint': True,
            'checkpoint_dir': 'checkpoints',
            'plot_training': True,  
            'plot_pca': True,
            'seed': 42,
            'device': 'auto'
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Dictionary containing configuration data.
        """
        if not self.config_path.exists():
            print(f"Config file not found, using defaults")
            return {'default': self._get_default_config()}

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f) or {}
                elif self.config_path.suffix == '.json':
                    data = json.load(f)
                else:
                    return {'default': self._get_default_config()}

            print(f"Loaded config from: {self.config_path}")
            return data

        except Exception as e:
            print(f"Error loading config: {e}")
            return {'default': self._get_default_config()}

    def get_config(self, dataset_name: str, verbose: bool = True) -> Dict[str, Any]:
        """Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset.
            verbose: Whether to print configuration source.

        Returns:
            Configuration dictionary for the dataset.
        """
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        config = self.default_config.copy()

        if dataset_name in self.configs and dataset_name != 'default':
            config.update(self.configs[dataset_name])
            if verbose:
                print(f"Using custom config for '{dataset_name}'")
        else:
            if verbose:
                print(f"Using default config for '{dataset_name}'")

        self._cache[dataset_name] = config
        return config

    def print_config(self, dataset_name: str) -> None:
        """Print configuration in a formatted manner.

        Args:
            dataset_name: Name of the dataset.
        """
        config = self.get_config(dataset_name, verbose=False)

        print("\nConfiguration: {}".format(dataset_name.upper()))
        print("-" * 70)

        categories = {
            'Model': ['num_prototypes', 'dropout', 'temperature'],
            'Training': ['batch_size', 'epochs', 'learning_rate', 'weight_decay'],
            'Regularization': ['label_smoothing', 'diversity_weight', 'grad_clip'],
            'Scheduling': ['early_stopping_patience', 'lr_scheduler'],
            'System': ['seed', 'device']
        }

        for category, keys in categories.items():
            print(f"\n[{category}]")
            for key in keys:
                if key in config:
                    print(f"  {key:25} : {config[key]}")

        print("-" * 70)
