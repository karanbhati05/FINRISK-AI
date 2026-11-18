"""
Tests for src.models.trainer.ModelTrainer.
"""

import numpy as np
import pandas as pd
import pytest


def _make_training_data(n: int = 200, seed: int = 42):
    """Return (X, y) with synthetic credit-like data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "LIMIT_BAL": rng.uniform(10000, 500000, n),
        "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([1, 2, 3, 4], n),
        "MARRIAGE": rng.choice([1, 2, 3], n),
        "AGE": rng.randint(21, 70, n),
        "PAY_0": rng.choice([-1, 0, 1, 2], n),
        "PAY_2": rng.choice([-1, 0, 1, 2], n),
        "PAY_3": rng.choice([-1, 0, 1], n),
        "PAY_4": rng.choice([-1, 0, 1], n),
        "PAY_5": rng.choice([-1, 0, 1], n),
        "PAY_6": rng.choice([-1, 0, 1], n),
        "BILL_AMT1": rng.uniform(0, 200000, n),
        "BILL_AMT2": rng.uniform(0, 200000, n),
        "BILL_AMT3": rng.uniform(0, 200000, n),
        "BILL_AMT4": rng.uniform(0, 200000, n),
        "BILL_AMT5": rng.uniform(0, 200000, n),
        "BILL_AMT6": rng.uniform(0, 200000, n),
        "PAY_AMT1": rng.uniform(0, 50000, n),
        "PAY_AMT2": rng.uniform(0, 50000, n),
        "PAY_AMT3": rng.uniform(0, 50000, n),
        "PAY_AMT4": rng.uniform(0, 50000, n),
        "PAY_AMT5": rng.uniform(0, 50000, n),
        "PAY_AMT6": rng.uniform(0, 50000, n),
    })
    y = rng.choice([0, 1], n, p=[0.78, 0.22])
    return X, y


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_import(self):
        """Module can be imported."""
        from src.models.trainer import ModelTrainer  # noqa: F401

    @pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
    def test_predict_output_shape(self, model_type):
        """predict() should return an array of length n."""
        from src.models.trainer import ModelTrainer

        X, y = _make_training_data(100)
        trainer = ModelTrainer(model_type=model_type, handle_imbalance="none")
        trainer.train(X, y)

        preds = trainer.predict(X)
        assert preds.shape == (100,), f"Expected (100,), got {preds.shape}"

    @pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
    def test_predict_proba_output_shape(self, model_type):
        """predict_proba() should return shape (n, 2) or (n,)."""
        from src.models.trainer import ModelTrainer

        X, y = _make_training_data(100)
        trainer = ModelTrainer(model_type=model_type, handle_imbalance="none")
        trainer.train(X, y)

        probas = trainer.predict_proba(X)
        assert probas.shape[0] == 100
        if probas.ndim == 2:
            assert probas.shape[1] == 2, "predict_proba should have 2 columns for binary classification"

    def test_predict_values_binary(self):
        """Predictions should be 0 or 1."""
        from src.models.trainer import ModelTrainer

        X, y = _make_training_data(100)
        trainer = ModelTrainer(model_type="logistic_regression", handle_imbalance="none")
        trainer.train(X, y)

        preds = trainer.predict(X)
        assert set(preds).issubset({0, 1}), f"Expected binary predictions, got {set(preds)}"

    def test_predict_proba_range(self):
        """Probabilities should be in [0, 1]."""
        from src.models.trainer import ModelTrainer

        X, y = _make_training_data(100)
        trainer = ModelTrainer(model_type="logistic_regression", handle_imbalance="none")
        trainer.train(X, y)

        probas = trainer.predict_proba(X)
        assert np.all(probas >= 0) and np.all(probas <= 1), "Probabilities must be in [0, 1]"
