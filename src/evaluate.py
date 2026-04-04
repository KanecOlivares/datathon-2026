"""Evaluation helpers."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score


def evaluate_accuracy(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return baseline accuracy metrics."""
    return {"accuracy": float(accuracy_score(y_true, y_pred))}
