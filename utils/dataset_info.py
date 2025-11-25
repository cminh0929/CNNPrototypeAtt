"""Utility functions for displaying dataset and model information."""
from typing import Dict, Any
from data.dataloader_manager import DataLoaderManager
from models.cnn_proto_attention import CNNProtoAttentionModel
from config.config_manager import ConfigManager


def show_dataset_info(dataset_name: str, config_manager: ConfigManager) -> None:
    """Display detailed information about a dataset without training.

    Args:
        dataset_name: Name of the dataset to show info for.
        config_manager: Configuration manager instance.
    """
    print(f"\nDataset Information: {dataset_name}")
    print("-" * 70)

    try:
        # Load configuration
        config = config_manager.get_config(dataset_name, verbose=False)
        
        # Load data to get info
        data_manager = DataLoaderManager(
            dataset_name=dataset_name,
            batch_size=config['batch_size'],
            use_augmentation=config.get('use_augmentation', False),
            augmentation_config=config.get('augmentation', {})
        )
        data_manager.load_and_prepare()
        info = data_manager.get_info()

        # Display dataset info
        print(f"\nDataset Type:    {'MULTIVARIATE' if info['input_channels'] > 1 else 'UNIVARIATE'}")
        print(f"Input Channels:  {info['input_channels']}")
        print(f"Time Steps:      {info['time_steps']}")
        print(f"Classes:         {info['num_classes']}")
        print(f"Train Samples:   {len(data_manager.X_train)}")
        print(f"Test Samples:    {len(data_manager.X_test)}")

        # Display configuration
        num_prototypes = config['num_prototypes']
        if num_prototypes is None:
            num_prototypes = info['num_classes'] * 2

        print(f"\nModel Configuration:")
        print(f"  Prototypes:    {num_prototypes}")
        print(f"  Dropout:       {config['dropout']}")
        print(f"  Temperature:   {config['temperature']}")
        
        # Display augmentation info
        use_augmentation = config.get('use_augmentation', False)
        print(f"\nData Augmentation:")
        print(f"  Enabled:       {'YES' if use_augmentation else 'NO'}")
        if use_augmentation:
            aug_config = config.get('augmentation', {})
            print(f"  Jitter Std:    {aug_config.get('jitter_std', 0.03)}")
            print(f"  Scaling Range: {aug_config.get('scaling_range', [0.8, 1.2])}")
            print(f"  Augment Prob:  {aug_config.get('augment_prob', 0.5)}")
        
        # Display projection info
        use_projection = config.get('use_projection', False)
        projection_dim = config.get('projection_dim', None)
        print(f"\nProjection Layer:")
        print(f"  Enabled:       {'YES' if use_projection else 'NO'}")
        if use_projection:
            if projection_dim is None:
                print(f"  Dimension:     Same as feature dim (256)")
            else:
                print(f"  Dimension:     {projection_dim}")
        
        # Display clustering loss info
        clustering_weight = config.get('clustering_weight', 0.0)
        print(f"\nClustering Loss:")
        print(f"  Enabled:       {'YES' if clustering_weight > 0 else 'NO'}")
        if clustering_weight > 0:
            print(f"  Weight:        {clustering_weight}")
            clustering_config = config.get('clustering_loss', {})
            print(f"  Compactness:   {clustering_config.get('compactness_weight', 1.0)}")
            print(f"  Separation:    {clustering_config.get('separation_weight', 0.5)}")

        print(f"\nTraining Configuration:")
        print(f"  Batch Size:    {config['batch_size']}")
        print(f"  Epochs:        {config['epochs']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Early Stop:    {config['early_stopping_patience']} epochs")

        # Estimate model size
        model = CNNProtoAttentionModel(
            input_channels=info['input_channels'],
            num_classes=info['num_classes'],
            num_prototypes=num_prototypes,
            dropout=config['dropout'],
            temperature=config['temperature'],
            use_projection=use_projection,
            projection_dim=projection_dim
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Size:")
        print(f"  Total Parameters:      {total_params:,}")
        print(f"  Trainable Parameters:  {trainable_params:,}")
        print(f"  Model Size (approx):   {total_params * 4 / 1024 / 1024:.2f} MB")

        print("-" * 70)

    except Exception as e:
        print(f"ERROR: Could not load dataset info: {str(e)}")
        import traceback
        traceback.print_exc()
