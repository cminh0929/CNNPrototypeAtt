from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, verbose: bool = True) -> float:
    """Evaluate model performance on a dataset.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader for evaluation data.
        device: Device to run evaluation on.
        verbose: Whether to print detailed results.

    Returns:
        Accuracy score.
    """
    model.eval()
    preds: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _, _ = model(X_batch)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(labels, preds)

    if verbose:
        print("\nEvaluation Results")
        print("-" * 70)
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"F1-Score: {f1_score(labels, preds, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds))
        print("-" * 70)

    return acc
