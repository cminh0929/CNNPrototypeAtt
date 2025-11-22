from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from models.cnn_proto_attention import CNNProtoAttentionModel
from data.dataloader_manager import DataLoaderManager
from training.trainer import Trainer
from training.evaluator import evaluate
from visualization.visualizer import Visualizer
from config.config_manager import ConfigManager
from utils.seed import set_seed
from utils.device import get_device


def main(dataset_name: str = "GunPoint") -> None:
    """Execute the complete training and evaluation pipeline.

    Args:
        dataset_name: Name of the UCR/UEA dataset to use.
    """
    config_manager: ConfigManager = ConfigManager("config.yaml")
    config: Dict[str, Any] = config_manager.get_config(dataset_name, verbose=True)
    config_manager.print_config(dataset_name)

    set_seed(config['seed'])
    print(f"Random seed: {config['seed']}")

    device: torch.device = get_device(config['device'])
    print(f"Device: {device}")

    data_manager: DataLoaderManager = DataLoaderManager(
        dataset_name=dataset_name,
        batch_size=config['batch_size']
    )
    data_manager.load_and_prepare()

    train_loader: DataLoader
    test_loader: DataLoader
    train_loader, test_loader = data_manager.get_loaders()
    info: Dict[str, int] = data_manager.get_info()

    num_prototypes: Optional[int] = config['num_prototypes']
    if num_prototypes is None:
        num_prototypes = info['num_classes'] * 2

    model: CNNProtoAttentionModel = CNNProtoAttentionModel(
        input_channels=info['input_channels'],
        num_classes=info['num_classes'],
        num_prototypes=num_prototypes,
        dropout=config['dropout'],
        temperature=config['temperature']
    ).to(device)

    total_params: int = sum(p.numel() for p in model.parameters())
    print(f"\n Model: {total_params:,} parameters")

    trainer: Trainer = Trainer(
        model=model,
        device=device,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        diversity_weight=config['diversity_weight'],
        label_smoothing=config['label_smoothing']
    )

    history: Dict[str, list] = trainer.train(
        train_loader,
        test_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )

    model.load_state_dict(trainer.best_model_state)

    final_acc: float = evaluate(model, test_loader, device, verbose=True)

    visualizer: Visualizer = Visualizer(model, device)
    
    # Generate all visualizations
    if config['plot_pca']:
        visualizer.plot_pca(test_loader, info['num_classes'])
    
    if config['plot_training']:
        visualizer.plot_training_curves(history)
    
    # New enhanced visualizations
    visualizer.plot_confusion_matrix(test_loader, info['num_classes'])
    visualizer.plot_prototype_heatmap()
    visualizer.plot_sample_predictions(test_loader, num_samples=6)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print('='*70)
    print(f"Dataset:        {dataset_name}")
    print(f"Type:           {'MULTIVARIATE' if info['input_channels'] > 1 else 'UNIVARIATE'}")
    print(f"Channels:       {info['input_channels']}")
    print(f"Classes:        {info['num_classes']}")
    print(f"Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print('='*70)


if __name__ == "__main__":
    main()
