"""
Evaluator module for CNNProto model evaluation.

Handles dataset evaluation, result saving, and standardized output format.
"""

from typing import Dict, Any
import json
from pathlib import Path

from data.dataloader_manager import DataLoaderManager
from config.config_manager import ConfigManager
from tsml.cnnproto_classifier import CNNProtoClassifier


def evaluate_with_tsml(dataset_name: str, config_manager: ConfigManager) -> None:
    """Evaluate CNNProto and save results in standardized format.

    Args:
        dataset_name: Name of the dataset to evaluate.
        config_manager: Configuration manager instance.
    """
    print(f"\nEvaluating CNNProto on {dataset_name}")
    print("-" * 70)

    try:
        # Import required metrics
        from sklearn.metrics import accuracy_score, classification_report
        
        # Note: We use our own DataLoaderManager instead of tsml for data loading
        # tsml-eval is optional and only used for standardized result format

        # Load data using DataLoaderManager
        data_manager = DataLoaderManager(dataset_name=dataset_name, batch_size=32)
        data_manager.load_and_prepare()

        X_train = data_manager.X_train
        y_train = data_manager.y_train
        X_test = data_manager.X_test
        y_test = data_manager.y_test

        # Get configuration
        config = config_manager.get_config(dataset_name, verbose=False)

        # Create and train classifier
        print("\nTraining CNNProto classifier...")
        clf = CNNProtoClassifier(
            num_prototypes=config['num_prototypes'],
            dropout=config['dropout'],
            temperature=config['temperature'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            diversity_weight=config['diversity_weight'],
            early_stopping_patience=config['early_stopping_patience'],
            seed=config['seed']
        )

        clf.fit(X_train, y_train, X_test, y_test)

        # Predict
        print("\nEvaluating on test set...")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nResults:")
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save results in standardized format (without large prediction arrays)
        results_dir = Path("results") / dataset_name / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "dataset": dataset_name,
            "classifier": "CNNProto",
            "accuracy": float(accuracy),
            "num_samples": len(y_test),
            "num_correct": int((y_pred == y_test).sum()),
            "config": config
        }

        with open(results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_dir / 'results.json'}")
        print("-" * 70)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def evaluate_without_tsml(dataset_name: str, config_manager: ConfigManager) -> None:
    """Evaluate CNNProto with basic output (no result file saved).

    Args:
        dataset_name: Name of the dataset to evaluate.
        config_manager: Configuration manager instance.
    """
    print("\nUsing basic evaluation (results not saved)...")

    from sklearn.metrics import accuracy_score, classification_report

    # Load data
    data_manager = DataLoaderManager(dataset_name=dataset_name, batch_size=32)
    data_manager.load_and_prepare()

    X_train = data_manager.X_train
    y_train = data_manager.y_train
    X_test = data_manager.X_test
    y_test = data_manager.y_test

    # Get configuration
    config = config_manager.get_config(dataset_name, verbose=False)

    # Create and train classifier
    print("\nTraining CNNProto classifier...")
    clf = CNNProtoClassifier(
        num_prototypes=config['num_prototypes'],
        dropout=config['dropout'],
        temperature=config['temperature'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        diversity_weight=config['diversity_weight'],
        early_stopping_patience=config['early_stopping_patience'],
        seed=config['seed']
    )

    clf.fit(X_train, y_train, X_test, y_test)

    # Predict
    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
