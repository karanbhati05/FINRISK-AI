"""
FinRisk AI - Data drift detection and model monitoring.

Uses the Kolmogorov-Smirnov test for numeric features and the chi-squared
test for categorical features to detect distribution shifts between a
reference (training) dataset and new incoming data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger("finrisk.monitoring")

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "monitoring"


class DriftDetector:
    """Detect data drift between a reference dataset and new observations."""

    def __init__(self, reference_data: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        reference_data : pd.DataFrame
            Training data used as the reference distribution.
        """
        self.reference_data = reference_data.copy()
        self.numeric_features = [
            c for c in reference_data.columns
            if c not in CATEGORICAL_FEATURES and reference_data[c].dtype in ("float64", "float32", "int64", "int32")
        ]
        self.categorical_features = [
            c for c in CATEGORICAL_FEATURES if c in reference_data.columns
        ]
        logger.info(
            "DriftDetector initialised with %d rows, %d numeric features, %d categorical features.",
            len(reference_data),
            len(self.numeric_features),
            len(self.categorical_features),
        )

    # ------------------------------------------------------------------
    # Core drift check
    # ------------------------------------------------------------------

    def check_drift(
        self,
        new_data: pd.DataFrame,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run statistical tests comparing *new_data* to the reference.

        Returns
        -------
        dict with keys:
            is_drifted, drift_score, drifted_features, per_feature_scores
        """
        per_feature: Dict[str, Dict[str, Any]] = {}
        drifted: List[str] = []

        # --- Numeric (KS test) ---
        for feat in self.numeric_features:
            if feat not in new_data.columns:
                continue
            ref_vals = self.reference_data[feat].dropna().values
            new_vals = new_data[feat].dropna().values
            if len(new_vals) == 0:
                continue
            stat, p_value = stats.ks_2samp(ref_vals, new_vals)
            per_feature[feat] = {
                "test": "ks",
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_value), 6),
                "is_drifted": p_value < threshold,
            }
            if p_value < threshold:
                drifted.append(feat)

        # --- Categorical (chi-squared test) ---
        for feat in self.categorical_features:
            if feat not in new_data.columns:
                continue
            ref_counts = self.reference_data[feat].value_counts().sort_index()
            new_counts = new_data[feat].value_counts().sort_index()
            # Align categories
            all_cats = sorted(set(ref_counts.index) | set(new_counts.index))
            ref_aligned = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
            new_aligned = np.array([new_counts.get(c, 0) for c in all_cats], dtype=float)
            # Normalize to expected frequencies
            if ref_aligned.sum() == 0 or new_aligned.sum() == 0:
                continue
            expected = ref_aligned / ref_aligned.sum() * new_aligned.sum()
            # Avoid zero expected
            expected = np.where(expected == 0, 1e-10, expected)
            stat, p_value = stats.chisquare(new_aligned, f_exp=expected)
            per_feature[feat] = {
                "test": "chi2",
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_value), 6),
                "is_drifted": p_value < threshold,
            }
            if p_value < threshold:
                drifted.append(feat)

        p_values = [v["p_value"] for v in per_feature.values()]
        drift_score = float(np.mean(p_values)) if p_values else 1.0

        return {
            "is_drifted": len(drifted) > 0,
            "drift_score": round(drift_score, 6),
            "drifted_features": drifted,
            "per_feature_scores": per_feature,
        }

    # ------------------------------------------------------------------
    # Prediction quality monitoring
    # ------------------------------------------------------------------

    def check_prediction_quality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_metrics: Dict[str, float],
        tolerance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Compare current metrics against *baseline_metrics* and flag degradation.

        Parameters
        ----------
        y_true : array-like   – true labels
        y_pred : array-like   – predicted labels (or probabilities for AUC)
        baseline_metrics : dict – {accuracy, f1, auc}
        tolerance : float – relative drop threshold (default 5 %)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # If y_pred looks like probabilities, threshold them for acc/f1
        y_pred_labels = (y_pred >= 0.5).astype(int) if y_pred.dtype == float else y_pred

        current: Dict[str, float] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred_labels)), 4),
            "f1": round(float(f1_score(y_true, y_pred_labels, zero_division=0)), 4),
        }
        try:
            current["auc"] = round(float(roc_auc_score(y_true, y_pred)), 4)
        except ValueError:
            current["auc"] = 0.0

        degraded: List[str] = []
        for metric, baseline_val in baseline_metrics.items():
            cur_val = current.get(metric)
            if cur_val is None:
                continue
            if baseline_val > 0 and (baseline_val - cur_val) / baseline_val > tolerance:
                degraded.append(metric)

        return {
            "is_degraded": len(degraded) > 0,
            "current_metrics": current,
            "baseline_metrics": baseline_metrics,
            "degraded_metrics": degraded,
        }

    # ------------------------------------------------------------------
    # Full monitoring report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        new_data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a full monitoring report (drift + optional quality check)
        and persist it to disk.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        drift = self.check_drift(new_data)

        report: Dict[str, Any] = {
            "report_timestamp": timestamp,
            "data_drift": drift,
            "is_drifted": drift["is_drifted"],
            "drift_score": drift["drift_score"],
            "drifted_features": drift["drifted_features"],
            "num_new_samples": len(new_data),
        }

        if predictions is not None and actuals is not None and baseline_metrics is not None:
            quality = self.check_prediction_quality(actuals, predictions, baseline_metrics)
            report["prediction_quality"] = quality

        # Persist
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_ts = timestamp.replace(":", "-").replace("+", "p")
        filepath = OUTPUTS_DIR / f"drift_report_{safe_ts}.json"
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Drift report saved to %s", filepath)

        return report

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_drift(
        self,
        new_data: pd.DataFrame,
        save_path: Optional[str] = None,
        top_n: int = 6,
    ) -> None:
        """
        Plot distribution comparisons for the top drifted features.
        """
        drift = self.check_drift(new_data)
        per_feat = drift["per_feature_scores"]
        # Sort by ascending p-value (most drifted first)
        sorted_feats = sorted(per_feat.items(), key=lambda x: x[1]["p_value"])[:top_n]
        if not sorted_feats:
            logger.info("No features to plot.")
            return

        n = len(sorted_feats)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (feat, info) in enumerate(sorted_feats):
            ax = axes[idx]
            ref_vals = self.reference_data[feat].dropna()
            new_vals = new_data[feat].dropna() if feat in new_data.columns else pd.Series(dtype=float)

            if info["test"] == "ks":
                ax.hist(ref_vals, bins=30, alpha=0.5, label="Reference", density=True)
                if len(new_vals):
                    ax.hist(new_vals, bins=30, alpha=0.5, label="New", density=True)
            else:
                ref_counts = ref_vals.value_counts(normalize=True).sort_index()
                new_counts = new_vals.value_counts(normalize=True).sort_index() if len(new_vals) else pd.Series(dtype=float)
                x = np.arange(len(ref_counts))
                width = 0.35
                ax.bar(x - width / 2, ref_counts.values, width, label="Reference", alpha=0.7)
                if len(new_counts):
                    ax.bar(x + width / 2, [new_counts.get(c, 0) for c in ref_counts.index], width, label="New", alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(ref_counts.index)

            drifted_flag = " ⚠" if info["is_drifted"] else ""
            ax.set_title(f"{feat} (p={info['p_value']:.4f}){drifted_flag}")
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(len(sorted_feats), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("Drift plot saved to %s", save_path)
        plt.close(fig)
