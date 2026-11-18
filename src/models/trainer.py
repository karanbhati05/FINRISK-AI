"""
FinRisk AI - Model Trainer

Trains credit default prediction models with imbalance handling and MLflow tracking.
Supports Logistic Regression, Random Forest, and XGBoost.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains a classification model for credit default prediction.

    Parameters
    ----------
    model_type : str
        One of 'logistic_regression', 'random_forest', 'xgboost'.
    handle_imbalance : str
        Strategy for class imbalance: 'smote', 'class_weight', or 'none'.
    random_state : int
        Random seed for reproducibility.
    """

    SUPPORTED_MODELS = {"logistic_regression", "random_forest", "xgboost"}
    SUPPORTED_IMBALANCE = {"smote", "class_weight", "none"}

    def __init__(
        self,
        model_type: str,
        handle_imbalance: str = "smote",
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_MODELS}, got '{model_type}'"
            )
        if handle_imbalance not in self.SUPPORTED_IMBALANCE:
            raise ValueError(
                f"handle_imbalance must be one of {self.SUPPORTED_IMBALANCE}, "
                f"got '{handle_imbalance}'"
            )

        self.model_type = model_type
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.model = None
        self.training_time: Optional[float] = None
        self.is_trained: bool = False
        self._classes: Optional[np.ndarray] = None

        self.model = self._build_model(model_type)
        logger.info(
            "ModelTrainer initialized: model_type=%s, handle_imbalance=%s",
            model_type,
            handle_imbalance,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, model_type: str):
        """Create a model instance with tuned default hyperparameters."""
        if model_type == "logistic_regression":
            params = dict(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=self.random_state,
            )
            if self.handle_imbalance == "class_weight":
                params["class_weight"] = "balanced"
            return LogisticRegression(**params)

        elif model_type == "random_forest":
            params = dict(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
            if self.handle_imbalance == "class_weight":
                params["class_weight"] = "balanced"
            return RandomForestClassifier(**params)

        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for the 'xgboost' model type. "
                    "Install it with: pip install xgboost"
                ) from exc

            params = dict(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1,
            )
            if self.handle_imbalance == "class_weight":
                # scale_pos_weight will be set at train time once we see y
                params["scale_pos_weight"] = 1  # placeholder
            return XGBClassifier(**params)

        raise ValueError(f"Unknown model_type: {model_type}")

    # ------------------------------------------------------------------
    # Imbalance handling
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_smote(
        X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Over-sample the minority class using SMOTE."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for SMOTE. "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE applied: %d -> %d samples", len(y_train), len(y_resampled)
        )
        return X_resampled, y_resampled

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "ModelTrainer":
        """
        Full training pipeline: apply imbalance handling, fit model, log to MLflow.

        Returns self for chaining.
        """
        X, y = np.asarray(X_train), np.asarray(y_train)
        self._classes = np.unique(y)

        # --- Imbalance handling ---
        if self.handle_imbalance == "smote":
            X, y = self._apply_smote(X, y)
        elif self.handle_imbalance == "class_weight" and self.model_type == "xgboost":
            # Set scale_pos_weight = #negative / #positive
            n_neg = int(np.sum(y == 0))
            n_pos = int(np.sum(y == 1))
            ratio = n_neg / max(n_pos, 1)
            self.model.set_params(scale_pos_weight=ratio)
            logger.info("XGBoost scale_pos_weight set to %.2f", ratio)

        # --- Train with MLflow tracking ---
        try:
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("handle_imbalance", self.handle_imbalance)
                mlflow.log_param("random_state", self.random_state)

                # Log model hyperparameters
                for k, v in self.model.get_params().items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        pass  # skip non-serializable params

                start = time.time()
                self.model.fit(X, y)
                self.training_time = time.time() - start

                mlflow.log_metric("training_time_seconds", self.training_time)
                mlflow.log_metric("training_samples", len(y))

                logger.info(
                    "Model trained in %.2fs (%d samples)",
                    self.training_time,
                    len(y),
                )
        except Exception as mlflow_err:
            # Fallback: train without MLflow if it's not configured
            logger.warning("MLflow logging failed (%s), training without it.", mlflow_err)
            start = time.time()
            self.model.fit(X, y)
            self.training_time = time.time() - start

        self.is_trained = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        self._check_trained()
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for each class."""
        self._check_trained()
        return self.model.predict_proba(np.asarray(X))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> str:
        """Save the trained model to disk using joblib."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to %s", path)
        return str(path)

    @classmethod
    def load_model(cls, path: str) -> "ModelTrainer":
        """Load a previously saved ModelTrainer from disk."""
        data = joblib.load(path)
        if isinstance(data, cls):
            logger.info("Model loaded from %s", path)
            return data
        elif isinstance(data, dict) and "model" in data:
            trainer = cls.__new__(cls)
            trainer.model = data["model"]
            trainer.model_type = data.get("model_type", "unknown")
            trainer.handle_imbalance = data.get("handle_imbalance", "none")
            trainer.random_state = data.get("random_state", 42)
            trainer.training_time = data.get("training_time")
            trainer._classes = data.get("classes")
            trainer.is_trained = True
            logger.info("Model loaded from dict in %s", path)
            return trainer
        raise TypeError(f"Unexpected object type {type(data)} in {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return a dict of the model's hyperparameters plus trainer metadata."""
        params = {
            "model_type": self.model_type,
            "handle_imbalance": self.handle_imbalance,
            "random_state": self.random_state,
            "training_time": self.training_time,
        }
        if self.model is not None:
            params.update(self.model.get_params())
        return params

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call .train() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"ModelTrainer(model_type='{self.model_type}', "
            f"handle_imbalance='{self.handle_imbalance}', {status})"
        )
