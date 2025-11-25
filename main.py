from typing import Dict, Any, Optional, List
import argparse
import sys
import time
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
from utils.results_manager import ResultsManager
from utils.dataset_utils import discover_datasets, validate_dataset, list_datasets


from utils.dataset_info import show_dataset_info


def run_single_dataset(
    dataset_name: str,
    config_manager: ConfigManager,
    results_manager: ResultsManager,
    save_results: bool = True
) -> Optional[Dict[str, Any]]:
    """Execute the complete training and evaluation pipeline for a single dataset.

    Args:
        dataset_name: Name of the UCR/UEA dataset to use.
        config_manager: Configuration manager instance.
        results_manager: Results manager instance.
        save_results: Whether to save results to disk.

    Returns:
        Dictionary containing experiment results, or None if failed.
    """
    print("\nRunning Experiment: {}".format(dataset_name))
    print("-" * 70)

    try:
        # Load configuration
        config: Dict[str, Any] = config_manager.get_config(dataset_name, verbose=True)
        config_manager.print_config(dataset_name)

        # Set seed
        set_seed(config['seed'])
        print(f"Random seed: {config['seed']}")

        # Get device
        device: torch.device = get_device(config['device'])
        print(f"Device: {device}")

        # Load data
        data_manager: DataLoaderManager = DataLoaderManager(
            dataset_name=dataset_name,
            batch_size=config['batch_size'],
            use_augmentation=config.get('use_augmentation', False),
            augmentation_config=config.get('augmentation', {})
        )
        data_manager.load_and_prepare()

        # --- ADAPTIVE BATCH SIZE (Important for small datasets) ---
        num_train_samples = len(data_manager.X_train)
        original_batch_size = config['batch_size']
        # Ensure minimum batch size of 8 for stable gradients
        # Use 1/10 of samples instead of 1/4 to have more batches per epoch
        adaptive_batch_size = max(8, min(original_batch_size, num_train_samples // 10))
        
        if adaptive_batch_size != original_batch_size:
            print(f"\n[Adaptive] Adjusting batch size: {original_batch_size} -> {adaptive_batch_size}")
            print(f"           Reason: Optimizing for dataset size ({num_train_samples} samples)")
            # Recreate data manager with adaptive batch size
            data_manager = DataLoaderManager(
                dataset_name=dataset_name,
                batch_size=adaptive_batch_size,
                use_augmentation=config.get('use_augmentation', False),
                augmentation_config=config.get('augmentation', {})
            )
            data_manager.load_and_prepare()

        train_loader: DataLoader
        test_loader: DataLoader
        train_loader, test_loader = data_manager.get_loaders()
        info: Dict[str, int] = data_manager.get_info()

        # Determine number of prototypes with dynamic heuristic
        num_prototypes: Optional[int] = config['num_prototypes']
        if num_prototypes is None:
            # Dynamic heuristic based on dataset complexity
            num_classes = info['num_classes']
            if num_classes <= 20:
                multiplier = 2  # Small/medium datasets -> Keep lightweight
            else:
                multiplier = 5  # Large/complex datasets -> Increase representation capacity
            
            num_prototypes = num_classes * multiplier
            print(f"\n[Auto-Config] Dynamic Prototypes: {num_prototypes} (Multiplier: {multiplier}x | Classes: {num_classes})")

        # Create model
        model: CNNProtoAttentionModel = CNNProtoAttentionModel(
            input_channels=info['input_channels'],
            num_classes=info['num_classes'],
            num_prototypes=num_prototypes,
            dropout=config['dropout'],
            temperature=config['temperature'],
            use_projection=config.get('use_projection', False),
            projection_dim=config.get('projection_dim', None)
        ).to(device)

        total_params: int = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {total_params:,} parameters")

        # Create trainer
        trainer: Trainer = Trainer(
            model=model,
            device=device,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            diversity_weight=config['diversity_weight'],
            label_smoothing=config['label_smoothing'],
            clustering_weight=config.get('clustering_weight', 0.0),
            clustering_config=config.get('clustering_loss', {})
        )
        if hasattr(model, 'initialize_prototypes'):
            print("\n[K-Means Init] Initializing prototypes based on training data...")
            # Call the prototype initialization method
            model.initialize_prototypes(train_loader, device=device)
        else:
            print("\n[WARNING] 'initialize_prototypes' method not found in model.")
            print("Model will use RANDOM initialization (Results may be poor).")
        # Train model
        start_time = time.time()
        history, best_epoch = trainer.train(
            train_loader,
            test_loader,
            epochs=config['epochs'],
            early_stopping_patience=config['early_stopping_patience']
        )
        training_time = time.time() - start_time

        # Load best model
        model.load_state_dict(trainer.best_model_state)

        # Final evaluation
        final_acc: float = evaluate(model, test_loader, device, verbose=True)

        # Generate visualizations in results directory
        visualizer: Visualizer = Visualizer(model, device)
        
        # Set current experiment context for visualization paths
        results_manager.set_current_experiment(dataset_name)
        
        if config['plot_pca']:
            pca_path = results_manager.get_visualization_path('pca_visualization.png')
            visualizer.plot_pca(test_loader, info['num_classes'], save_path=pca_path)
        
        if config['plot_training']:
            training_path = results_manager.get_visualization_path('training_curves.png')
            visualizer.plot_training_curves(history, save_path=training_path)
        
        cm_path = results_manager.get_visualization_path('confusion_matrix.png')
        visualizer.plot_confusion_matrix(test_loader, info['num_classes'], save_path=cm_path)
        
        heatmap_path = results_manager.get_visualization_path('prototype_heatmap.png')
        visualizer.plot_prototype_heatmap(save_path=heatmap_path)
        
        samples_path = results_manager.get_visualization_path('sample_predictions.png')
        visualizer.plot_sample_predictions(test_loader, num_samples=6, save_path=samples_path)

        # Print summary
        print("\nFinal Summary")
        print("-" * 70)
        print(f"Dataset:        {dataset_name}")
        print(f"Type:           {'MULTIVARIATE' if info['input_channels'] > 1 else 'UNIVARIATE'}")
        print(f"Channels:       {info['input_channels']}")
        print(f"Classes:        {info['num_classes']}")
        print(f"Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
        print(f"Training time:  {training_time:.2f}s")
        print("-" * 70)

        # Prepare result data
        result_data = {
            "dataset_name": dataset_name,
            "config": config,
            "dataset_info": {
                "input_channels": info['input_channels'],
                "num_classes": info['num_classes'],
                "time_steps": info['time_steps'],
                "train_size": len(train_loader.dataset),
                "test_size": len(test_loader.dataset)
            },
            "training_history": history,
            "final_metrics": {
                "test_accuracy": final_acc,
                "best_epoch": best_epoch,
                "total_epochs": len(history['loss']),
                "training_time": training_time,
                "total_parameters": total_params
            }
        }

        # Save results and check if this is the best result
        if save_results:
            _, is_best = results_manager.save_result(
                dataset_name=dataset_name,
                config=config,
                dataset_info=result_data["dataset_info"],
                history=history,
                final_metrics=result_data["final_metrics"]
            )
            
            # If this is the best result, copy visualizations to best folder
            if is_best:
                results_manager.copy_visualizations_to_best(dataset_name)

        return result_data

    except Exception as e:
        print(f"\nERROR: Failed to run experiment on {dataset_name}")
        print(f"Reason: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_all_datasets(
    config_manager: ConfigManager,
    results_manager: ResultsManager
) -> None:
    """Run experiments on all available datasets.

    Args:
        config_manager: Configuration manager instance.
        results_manager: Results manager instance.
    """
    # Discover datasets
    datasets = discover_datasets()

    if not datasets:
        print("\nNo valid datasets found!")
        print("Please add datasets to the 'datasets/' directory.")
        print("See datasets/README.md for instructions.")
        return

    print("\nRunning Experiments on All Datasets ({} total)".format(len(datasets)))
    print("-" * 70)

    all_results: List[Dict[str, Any]] = []

    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing: {dataset_name}")
        
        result = run_single_dataset(
            dataset_name=dataset_name,
            config_manager=config_manager,
            results_manager=results_manager,
            save_results=True
        )

        if result is not None:
            all_results.append(result)

    # Generate and print summary
    if all_results:
        results_manager.print_summary(all_results)
        results_manager.generate_summary(all_results)
    else:
        print("\nNo successful experiments to summarize.")


def main() -> None:
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="CNN Prototype Attention Model for Time Series Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Run on default dataset (GunPoint)
  python main.py --dataset ECG200     # Run on specific dataset
  python main.py --all                # Run on all available datasets
  python main.py --list               # List available datasets
  python main.py --info               # Show info for all datasets
  python main.py --dataset ECG200 --info  # Show info for specific dataset
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='GunPoint',
        help='Name of the dataset to run (default: GunPoint)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run experiments on all available datasets'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets and exit'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show detailed information about datasets without training'
    )

    args = parser.parse_args()

    # Initialize managers
    config_manager = ConfigManager("config.yaml")
    results_manager = ResultsManager("results")

    # Handle --list flag
    if args.list:
        list_datasets(verbose=True)
        sys.exit(0)

    # Handle --info flag
    if args.info:
        if args.all:
            # Show info for all datasets
            datasets = discover_datasets()
            if not datasets:
                print("\nNo valid datasets found!")
                sys.exit(1)
            
            for dataset_name in datasets:
                show_dataset_info(dataset_name, config_manager)
        else:
            # Show info for specific dataset
            dataset_name = args.dataset
            if not validate_dataset(dataset_name):
                print(f"\nERROR: Dataset '{dataset_name}' not found or invalid!")
                print("\nAvailable datasets:")
                list_datasets(verbose=True)
                sys.exit(1)
            show_dataset_info(dataset_name, config_manager)
        sys.exit(0)

    # Handle --all flag
    if args.all:
        run_all_datasets(config_manager, results_manager)
        sys.exit(0)

    # Run single dataset
    dataset_name = args.dataset

    # Validate dataset exists
    if not validate_dataset(dataset_name):
        print(f"\nERROR: Dataset '{dataset_name}' not found or invalid!")
        print("\nAvailable datasets:")
        list_datasets(verbose=True)
        sys.exit(1)

    # Run experiment
    run_single_dataset(
        dataset_name=dataset_name,
        config_manager=config_manager,
        results_manager=results_manager,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
