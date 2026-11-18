"""
FinRisk AI - Model Evaluator

Evaluates, compares, and visualises trained credit-default models.
Generates ROC curves, Precision-Recall curves, confusion matrices,
threshold-tuning plots, and logs everything to MLflow.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

try:
    import mlflow
except ImportError:
    mlflow = None

from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

# Default output directory for plots
DEFAULT_PLOT_DIR = Path("outputs/model_plots")


class ModelEvaluator:
    """
    Evaluate and compare multiple trained ModelTrainer instances.

    Parameters
    ----------
    models : dict[str, ModelTrainer]
        Mapping of human-readable name → trained ModelTrainer.
    """

    def __init__(self, models: Dict[str, ModelTrainer]):
        if not models:
            raise ValueError("At least one model must be provided.")
        for name, m in models.items():
            if not m.is_trained:
                raise ValueError(f"Model '{name}' has not been trained yet.")
        self.models = models
        self._results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Compute comprehensive metrics for every model.

        Returns a dict of ``{model_name: {metric_name: value, ...}}``.
        """
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        results: Dict[str, Dict] = {}

        for name, trainer in self.models.items():
            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test)[:, 1]

            metrics: Dict = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "precision_weighted": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_weighted": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "pr_auc": average_precision_score(y_test, y_proba),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(
                    y_test, y_pred, zero_division=0
                ),
            }

            # Log to MLflow
            self._log_metrics_to_mlflow(name, metrics)

            results[name] = metrics
            logger.info(
                "Evaluated '%s': ROC-AUC=%.4f  F1-macro=%.4f",
                name,
                metrics["roc_auc"],
                metrics["f1_macro"],
            )

        self._results = results
        return results

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Return a side-by-side DataFrame comparing scalar metrics for all models.
        """
        if not self._results:
            self.evaluate(X_test, y_test)

        scalar_keys = [
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
            "roc_auc",
            "pr_auc",
        ]
        rows = {}
        for name, metrics in self._results.items():
            rows[name] = {k: round(metrics[k], 4) for k in scalar_keys}

        df = pd.DataFrame(rows).T
        df.index.name = "model"
        return df

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot ROC curves for all models on a single figure."""
        save_path = self._resolve_path(save_path, "roc_curves.png")
        X_test, y_test = np.asarray(X_test), np.asarray(y_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, trainer in self.models.items():
            y_proba = trainer.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Model Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("ROC curves saved to %s", save_path)
        return str(save_path)

    def plot_precision_recall_curves(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot Precision-Recall curves for all models on a single figure."""
        save_path = self._resolve_path(save_path, "precision_recall_curves.png")
        X_test, y_test = np.asarray(X_test), np.asarray(y_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, trainer in self.models.items():
            y_proba = trainer.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            ax.plot(recall, precision, label=f"{name} (AP={pr_auc:.4f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves — Model Comparison")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("PR curves saved to %s", save_path)
        return str(save_path)

    def plot_confusion_matrices(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot a grid of confusion matrices — one per model."""
        save_path = self._resolve_path(save_path, "confusion_matrices.png")
        X_test, y_test = np.asarray(X_test), np.asarray(y_test)
        n = len(self.models)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (name, trainer) in enumerate(self.models.items()):
            y_pred = trainer.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["No Default", "Default"]
            )
            disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
            axes[idx].set_title(name, fontsize=11)

        # Hide unused axes
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrices saved to %s", save_path)
        return str(save_path)

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def tune_threshold(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric: str = "f1",
        save_path: Optional[str] = None,
    ) -> float:
        """
        Find the optimal classification threshold for *model_name* by scanning
        from 0.10 to 0.90 (step 0.01), optimising for *metric*.

        Returns the optimal threshold and saves a threshold-vs-metrics plot.
        """
        if model_name not in self.models:
            raise KeyError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.models.keys())}"
            )
        metric = metric.lower()
        if metric not in ("f1", "recall", "precision"):
            raise ValueError("metric must be 'f1', 'recall', or 'precision'")

        trainer = self.models[model_name]
        X_test, y_test = np.asarray(X_test), np.asarray(y_test)
        y_proba = trainer.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.10, 0.91, 0.01)
        f1_scores: List[float] = []
        recall_scores: List[float] = []
        precision_scores: List[float] = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
            precision_scores.append(
                precision_score(y_test, y_pred, zero_division=0)
            )

        metric_map = {
            "f1": f1_scores,
            "recall": recall_scores,
            "precision": precision_scores,
        }
        target_values = metric_map[metric]
        best_idx = int(np.argmax(target_values))
        optimal_threshold = float(thresholds[best_idx])

        logger.info(
            "Optimal threshold for '%s' (metric=%s): %.2f  (score=%.4f)",
            model_name,
            metric,
            optimal_threshold,
            target_values[best_idx],
        )

        # --- Plot ---
        save_path = self._resolve_path(
            save_path, f"threshold_tuning_{model_name}.png"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, f1_scores, label="F1")
        ax.plot(thresholds, recall_scores, label="Recall")
        ax.plot(thresholds, precision_scores, label="Precision")
        ax.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal ({metric})={optimal_threshold:.2f}",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold Tuning — {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Threshold plot saved to %s", save_path)

        return optimal_threshold

    # ------------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------------

    def get_best_model(self, metric: str = "roc_auc") -> str:
        """
        Return the name of the best model by the specified scalar metric.

        ``evaluate()`` must have been called first.
        """
        if not self._results:
            raise RuntimeError("Call evaluate() before get_best_model().")
        if metric not in self._results[next(iter(self._results))]:
            raise KeyError(f"Unknown metric '{metric}'")

        best_name = max(
            self._results, key=lambda n: self._results[n][metric]
        )
        logger.info(
            "Best model by %s: '%s' (%.4f)",
            metric,
            best_name,
            self._results[best_name][metric],
        )
        return best_name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(user_path: Optional[str], default_name: str) -> Path:
        """Resolve the save path, creating directories as needed."""
        if user_path:
            p = Path(user_path)
        else:
            p = DEFAULT_PLOT_DIR / default_name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _log_metrics_to_mlflow(name: str, metrics: Dict) -> None:
        """Best-effort log of scalar metrics to MLflow."""
        if mlflow is None:
            return
        try:
            with mlflow.start_run(nested=True, run_name=f"eval-{name}"):
                for key in (
                    "accuracy",
                    "precision_macro",
                    "precision_weighted",
                    "recall_macro",
                    "recall_weighted",
                    "f1_macro",
                    "f1_weighted",
                    "roc_auc",
                    "pr_auc",
                ):
                    mlflow.log_metric(f"{name}_{key}", metrics[key])
        except Exception as exc:
            logger.warning("MLflow metric logging failed for '%s': %s", name, exc)

    def __repr__(self) -> str:
        return f"ModelEvaluator(models={list(self.models.keys())})"
