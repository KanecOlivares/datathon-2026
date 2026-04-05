"""Training scaffold."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier

from src.config import RANDOM_SEED


def build_baseline_model() -> DummyClassifier:
    """Create a simple baseline classifier."""
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
