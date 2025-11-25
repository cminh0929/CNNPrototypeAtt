import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil


class ResultsManager:
    """Manages saving and loading of experiment results with best model tracking."""

    def __init__(self, results_dir: str = "results") -> None:
        """Initialize the ResultsManager.

        Args:
            results_dir: Base directory for storing results.
        """
        self.results_dir: Path = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.current_dataset: Optional[str] = None
        self.current_timestamp: Optional[str] = None

    def get_dataset_dir(self, dataset_name: str) -> Path:
        """Get the directory path for a specific dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Path to the dataset directory.
        """
        dataset_dir = self.results_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        return dataset_dir

    def get_visualization_dir(self, dataset_name: str, subdir: str = "current") -> Path:
        """Get the visualization directory for a specific dataset.

        Args:
            dataset_name: Name of the dataset.
            subdir: Subdirectory name ('current' or 'best').

        Returns:
            Path to the visualization directory.
        """
        viz_dir = self.get_dataset_dir(dataset_name) / "visualizations" / subdir
        viz_dir.mkdir(parents=True, exist_ok=True)
        return viz_dir

    def set_current_experiment(self, dataset_name: str, timestamp: Optional[str] = None) -> None:
        """Set the current experiment context for saving visualizations.

        Args:
            dataset_name: Name of the dataset.
            timestamp: Optional timestamp string, auto-generated if None.
        """
        self.current_dataset = dataset_name
        self.current_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_visualization_path(self, filename: str, is_best: bool = False) -> str:
        """Get the full path for a visualization file in the current experiment.

        Args:
            filename: Name of the visualization file.
            is_best: If True, save to 'best' directory, otherwise 'current'.

        Returns:
            Full path string for the visualization file.
        """
        if self.current_dataset is None:
            return filename
        
        subdir = "best" if is_best else "current"
        viz_dir = self.get_visualization_dir(self.current_dataset, subdir)
        return str(viz_dir / filename)

    def load_best_result(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load the best result for a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Best result dictionary or None if not found.
        """
        best_path = self.get_dataset_dir(dataset_name) / "best.json"
        if not best_path.exists():
            return None
        
        with open(best_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_history(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load the run history for a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            List of run history entries.
        """
        history_path = self.get_dataset_dir(dataset_name) / "history.json"
        if not history_path.exists():
            return []
        
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_history(self, dataset_name: str, run_info: Dict[str, Any]) -> None:
        """Update the run history for a dataset.

        Args:
            dataset_name: Name of the dataset.
            run_info: Information about the current run.
        """
        history = self.load_history(dataset_name)
        history.append(run_info)
        
        history_path = self.get_dataset_dir(dataset_name) / "history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def save_result(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        dataset_info: Dict[str, int],
        history: Dict[str, List[float]],
        final_metrics: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> str:
        """Save experiment results with best model tracking.

        Args:
            dataset_name: Name of the dataset.
            config: Configuration dictionary.
            dataset_info: Dataset information (channels, classes, etc.).
            history: Training history (loss, accuracy per epoch).
            final_metrics: Final evaluation metrics.
            timestamp: Optional timestamp string, auto-generated if None.

        Returns:
            Path to the saved result file.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.set_current_experiment(dataset_name, timestamp)
        dataset_dir = self.get_dataset_dir(dataset_name)

        current_acc = final_metrics["test_accuracy"]

        # Calculate training summary statistics
        best_train_acc = max(history["train_acc"])
        best_test_acc = max(history["test_acc"])
        final_train_acc = history["train_acc"][-1]
        final_test_acc = history["test_acc"][-1]
        min_loss = min(history["loss"])
        final_loss = history["loss"][-1]

        # Prepare simplified result data
        result_data = {
            "run_info": {
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "run_id": timestamp
            },
            "performance": {
                "test_accuracy": final_metrics["test_accuracy"],
                "best_epoch": final_metrics.get("best_epoch", final_metrics["total_epochs"]),
                "total_epochs": final_metrics["total_epochs"],
                "training_time_seconds": final_metrics["training_time"],
                "model_parameters": final_metrics["total_parameters"]
            },
            "dataset_info": {
                "type": "MULTIVARIATE" if dataset_info["input_channels"] > 1 else "UNIVARIATE",
                "channels": dataset_info["input_channels"],
                "classes": dataset_info["num_classes"],
                "time_steps": dataset_info["time_steps"],
                "train_samples": dataset_info["train_size"],
                "test_samples": dataset_info["test_size"]
            },
            "hyperparameters": {
                "num_prototypes": config["num_prototypes"],
                "dropout": config["dropout"],
                "temperature": config["temperature"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "diversity_weight": config["diversity_weight"],
                # Data augmentation
                "use_augmentation": config.get("use_augmentation", False),
                "augmentation_config": config.get("augmentation", {}) if config.get("use_augmentation", False) else None,
                # Projection layer
                "use_projection": config.get("use_projection", False),
                "projection_dim": config.get("projection_dim", None) if config.get("use_projection", False) else None,
                # Clustering loss
                "clustering_weight": config.get("clustering_weight", 0.0),
                "clustering_config": config.get("clustering_loss", {}) if config.get("clustering_weight", 0.0) > 0 else None
            },
            "training_summary": {
                "loss": {
                    "min": round(min_loss, 4),
                    "final": round(final_loss, 4)
                },
                "train_accuracy": {
                    "best": round(best_train_acc, 4),
                    "final": round(final_train_acc, 4)
                },
                "test_accuracy": {
                    "best": round(best_test_acc, 4),
                    "final": round(final_test_acc, 4)
                },
                "generalization_gap": {
                    "at_best_epoch": round(best_train_acc - best_test_acc, 4),
                    "final": round(final_train_acc - final_test_acc, 4)
                }
            }
        }

        # Save current result
        current_path = dataset_dir / "current.json"
        with open(current_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)

        # Check if this is the best result
        best_result = self.load_best_result(dataset_name)
        is_best = False
        
        if best_result is None or current_acc > best_result["performance"]["test_accuracy"]:
            is_best = True
            best_path = dataset_dir / "best.json"
            with open(best_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"\nNEW BEST RESULT! Accuracy: {current_acc:.4f} ({current_acc*100:.2f}%)")
            if best_result:
                prev_best = best_result["performance"]["test_accuracy"]
                improvement = (current_acc - prev_best) * 100
                print(f"Improvement: +{improvement:.2f}% (previous best: {prev_best:.4f})")
        else:
            prev_best = best_result["performance"]["test_accuracy"]
            print(f"\nCurrent accuracy: {current_acc:.4f} ({current_acc*100:.2f}%)")
            print(f"Best accuracy: {prev_best:.4f} ({prev_best*100:.2f}%)")

        # Update run history
        run_summary = {
            "timestamp": datetime.now().isoformat(),
            "run_id": timestamp,
            "accuracy": current_acc,
            "epochs": final_metrics["total_epochs"],
            "training_time": final_metrics["training_time"],
            "is_best": is_best
        }
        self.update_history(dataset_name, run_summary)

        # Print save locations
        print(f"\nResults saved:")
        print(f"  Current: {current_path}")
        if is_best:
            print(f"  Best: {dataset_dir / 'best.json'}")
        print(f"  History: {dataset_dir / 'history.json'}")
        print(f"  Visualizations: {self.get_visualization_dir(dataset_name, 'current')}")
        if is_best:
            print(f"  Best visualizations: {self.get_visualization_dir(dataset_name, 'best')}")

        return str(current_path), is_best

    def copy_visualizations_to_best(self, dataset_name: str) -> None:
        """Copy current visualizations to best directory.

        Args:
            dataset_name: Name of the dataset.
        """
        current_dir = self.get_visualization_dir(dataset_name, "current")
        best_dir = self.get_visualization_dir(dataset_name, "best")

        # Copy all PNG files from current to best
        for file in current_dir.glob("*.png"):
            shutil.copy2(file, best_dir / file.name)



    def generate_summary(
        self,
        dataset_results: List[Dict[str, Any]],
        timestamp: Optional[str] = None
    ) -> str:
        """Generate a summary report across multiple datasets.

        Args:
            dataset_results: List of result dictionaries from multiple datasets.
            timestamp: Optional timestamp string, auto-generated if None.

        Returns:
            Path to the saved summary file.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(dataset_results),
            "datasets": {}
        }

        for result in dataset_results:
            dataset_name = result["dataset_name"]
            summary["datasets"][dataset_name] = {
                "test_accuracy": result["final_metrics"]["test_accuracy"],
                "training_time": result["final_metrics"]["training_time"],
                "total_epochs": result["final_metrics"]["total_epochs"],
                "num_classes": result["dataset_info"]["num_classes"],
                "input_channels": result["dataset_info"]["input_channels"]
            }

        accuracies = [r["final_metrics"]["test_accuracy"] for r in dataset_results]
        summary["aggregate"] = {
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "max_accuracy": max(accuracies) if accuracies else 0,
            "min_accuracy": min(accuracies) if accuracies else 0,
            "total_training_time": sum(r["final_metrics"]["training_time"] for r in dataset_results)
        }

        filename = f"summary_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {filepath}")
        return str(filepath)

    def print_summary(self, dataset_results: List[Dict[str, Any]]) -> None:
        """Print a formatted summary to console.

        Args:
            dataset_results: List of result dictionaries from multiple datasets.
        """
        if not dataset_results:
            print("No results to summarize.")
            return

        print("\nExperiment Summary")
        print("-" * 80)
        print(f"Total datasets: {len(dataset_results)}")
        print(f"\n{'Dataset':<20} {'Accuracy':<12} {'Epochs':<10} {'Time (s)':<12} {'Classes':<10}")
        print("-" * 80)

        total_time = 0
        accuracies = []

        for result in dataset_results:
            name = result["dataset_name"]
            acc = result["final_metrics"]["test_accuracy"]
            epochs = result["final_metrics"]["total_epochs"]
            time_s = result["final_metrics"]["training_time"]
            classes = result["dataset_info"]["num_classes"]

            accuracies.append(acc)
            total_time += time_s

            print(f"{name:<20} {acc:.4f} ({acc*100:5.2f}%) {epochs:<10} {time_s:<12.2f} {classes:<10}")

        print("-" * 80)
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"{'AVERAGE':<20} {avg_acc:.4f} ({avg_acc*100:5.2f}%)")
        print(f"{'TOTAL TIME':<20} {'':<12} {'':<10} {total_time:<12.2f}")
        print("-" * 80)
