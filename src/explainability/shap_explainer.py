"""
SHAP-based model explainability for FinRisk AI.

Provides local and global explanations for credit default predictions
using SHAP (SHapley Additive exPlanations).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    """Wrapper around SHAP for explaining credit default models.

    Supports LogisticRegression (LinearExplainer), RandomForest and
    XGBoost (TreeExplainer).
    """

    SUPPORTED_TYPES = {"logistic_regression", "random_forest", "xgboost"}

    def __init__(self, model: Any, model_type: str, feature_names: List[str]):
        """
        Parameters
        ----------
        model : trained sklearn / xgboost model object
        model_type : one of 'logistic_regression', 'random_forest', 'xgboost'
        feature_names : list of feature names matching the model's input columns
        """
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_TYPES}, got '{model_type}'"
            )
        self.model = model
        self.model_type = model_type
        self.feature_names = list(feature_names)
        self.explainer = self._create_explainer()
        self.shap_values: Optional[np.ndarray] = None
        self.X_sample: Optional[np.ndarray] = None
        self.base_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Explainer factory
    # ------------------------------------------------------------------
    def _create_explainer(self) -> Any:
        """Create the appropriate SHAP explainer for the model type."""
        if self.model_type == "logistic_regression":
            # LinearExplainer needs a background dataset; defer to compute_shap_values
            return None
        elif self.model_type in ("random_forest", "xgboost"):
            return shap.TreeExplainer(self.model)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _extract_binary_shap(self, raw: Any) -> np.ndarray:
        """Extract SHAP values for the positive/default class (class 1)."""
        if isinstance(raw, list):
            arr = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 3 and raw.shape[2] == 2:
                # Shape (n_samples, n_features, 2) from RF TreeExplainer
                arr = raw[:, :, 1]
            elif raw.ndim == 3 and raw.shape[2] == 1:
                arr = raw[:, :, 0]
            else:
                arr = raw
        else:
            arr = np.asarray(raw)
        return arr

    def _extract_base_value(self) -> float:
        """Extract scalar base value / expected value for class 1."""
        if self.explainer is None:
            return 0.0
        bv = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(bv, (list, np.ndarray)):
            if len(bv) == 2:
                return float(bv[1])
            elif len(bv) >= 1:
                return float(bv[0])
            else:
                return 0.0
        return float(bv)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute_shap_values(
        self, X: Union[np.ndarray, pd.DataFrame], max_samples: int = 500
    ) -> np.ndarray:
        """Compute SHAP values, sampling if the dataset is large.

        Parameters
        ----------
        X : feature matrix (n_samples, n_features)
        max_samples : maximum rows to use (randomly sampled)

        Returns
        -------
        shap_values : array of shape (n_sampled, n_features)
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Sub-sample if needed
        if X_arr.shape[0] > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_arr.shape[0], size=max_samples, replace=False)
            X_arr = X_arr[idx]

        self.X_sample = X_arr

        # For linear models we need a background dataset
        if self.model_type == "logistic_regression":
            self.explainer = shap.LinearExplainer(self.model, X_arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self.explainer.shap_values(X_arr)

        self.shap_values = self._extract_binary_shap(raw)
        self.base_value = self._extract_base_value()

        return self.shap_values

    # ------------------------------------------------------------------
    # Global plots
    # ------------------------------------------------------------------
    def plot_summary(self, save_path: Union[str, Path]) -> None:
        """SHAP beeswarm summary plot."""
        self._ensure_computed()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(
        self, save_path: Union[str, Path], top_n: int = 15
    ) -> None:
        """Bar chart of mean |SHAP| for top-N features."""
        self._ensure_computed()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 7))
        names = [self.feature_names[int(i)] for i in indices]
        values = [float(mean_abs[int(i)]) for i in indices]
        ax.barh(range(len(names)), values[::-1], color="#1f77b4")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_dependence(
        self,
        feature_name: str,
        save_path: Union[str, Path],
        interaction_feature: Optional[str] = None,
    ) -> None:
        """SHAP dependence plot for a specific feature."""
        self._ensure_computed()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        interaction = interaction_feature if interaction_feature else "auto"
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            pd.DataFrame(self.X_sample, columns=self.feature_names),
            interaction_index=interaction,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Individual prediction explanation
    # ------------------------------------------------------------------
    def explain_prediction(
        self,
        X_single: Union[np.ndarray, pd.DataFrame],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Explain a single customer prediction.

        Parameters
        ----------
        X_single : 1-row feature matrix
        save_path : optional path to save waterfall plot

        Returns
        -------
        dict with:
            predicted_probability: float
            base_value: float
            top_positive_factors: list of (feature, value, shap_value)
            top_negative_factors: list of (feature, value, shap_value)
            top_risk_factors: list of dicts for API schemas
            top_protective_factors: list of dicts for API schemas
            risk_explanation: human-readable text
        """
        if isinstance(X_single, pd.DataFrame):
            x_arr = X_single.values
        else:
            x_arr = np.asarray(X_single)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        # Ensure linear explainer is initialized
        if self.model_type == "logistic_regression" and self.explainer is None:
            self.explainer = shap.LinearExplainer(self.model, x_arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self.explainer.shap_values(x_arr)

        if isinstance(raw, list):
            sv = np.asarray(raw[1] if len(raw) > 1 else raw[0]).flatten()
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                sv = raw[0, :, 1].flatten() if raw.shape[2] == 2 else raw[0, :, 0].flatten()
            elif raw.ndim == 2:
                sv = raw[0, :].flatten()
            else:
                sv = raw.flatten()
        else:
            sv = np.asarray(raw).flatten()

        base_val = self._extract_base_value()

        # Predicted probability
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(x_arr)[0, 1])
        else:
            prob = float(1 / (1 + np.exp(-(base_val + sv.sum()))))

        # Sort features by SHAP magnitude
        order = np.argsort(np.abs(sv))[::-1]

        top_positive_tuples: List[Tuple[str, float, float]] = []
        top_negative_tuples: List[Tuple[str, float, float]] = []
        top_risk_factors_dicts: List[Dict[str, Any]] = []
        top_protective_dicts: List[Dict[str, Any]] = []

        for idx in order:
            i = int(idx)
            feat = self.feature_names[i]
            val = float(x_arr[0, i])
            shap_val = float(sv[i])
            if shap_val > 0:
                top_positive_tuples.append((feat, val, shap_val))
                top_risk_factors_dicts.append(
                    {
                        "feature": feat,
                        "value": round(val, 4),
                        "shap_value": round(shap_val, 4),
                        "direction": "positive",
                    }
                )
            elif shap_val < 0:
                top_negative_tuples.append((feat, val, shap_val))
                top_protective_dicts.append(
                    {
                        "feature": feat,
                        "value": round(val, 4),
                        "shap_value": round(shap_val, 4),
                        "direction": "negative",
                    }
                )

        # Cap lists for display
        top_positive = top_positive_tuples[:10]
        top_negative = top_negative_tuples[:10]
        top_risk_factors_dicts = top_risk_factors_dicts[:10]
        top_protective_dicts = top_protective_dicts[:10]

        # Build human-readable explanation
        risk_label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        explanation_parts = [
            f"This customer is classified as {risk_label} "
            f"(probability: {prob:.2f})."
        ]
        if top_positive:
            risk_factors = ", ".join(
                f"{f}={v:.2f} (contributing +{s:.4f})"
                for f, v, s in top_positive[:3]
            )
            explanation_parts.append(f"The main risk factors are: {risk_factors}.")
        if top_negative:
            mitigating = ", ".join(
                f"{f}={v:.2f} ({s:.4f})" for f, v, s in top_negative[:3]
            )
            explanation_parts.append(f"Mitigating factors include: {mitigating}.")

        risk_explanation = " ".join(explanation_parts)

        # Waterfall plot
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                expl = shap.Explanation(
                    values=sv,
                    base_values=base_val,
                    data=x_arr.flatten(),
                    feature_names=self.feature_names,
                )
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(expl, show=False)
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                # Fallback: manual bar chart
                self._fallback_waterfall(sv, x_arr.flatten(), base_val, save_path)

        return {
            "predicted_probability": round(prob, 4),
            "base_value": round(base_val, 4),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "top_risk_factors": top_risk_factors_dicts,
            "top_protective_factors": top_protective_dicts,
            "shap_values": {
                self.feature_names[i]: round(float(sv[i]), 4)
                for i in range(len(self.feature_names))
            },
            "risk_explanation": risk_explanation,
        }

    # ------------------------------------------------------------------
    # Risk factors summary
    # ------------------------------------------------------------------
    def get_top_risk_factors(self, n: int = 10) -> pd.DataFrame:
        """Top N features by mean |SHAP|.

        Returns
        -------
        DataFrame with columns: feature, mean_abs_shap, direction
        """
        self._ensure_computed()
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        mean_signed = self.shap_values.mean(axis=0)
        indices = np.argsort(mean_abs)[::-1][:n]

        records = []
        for i in indices:
            idx = int(i)
            records.append(
                {
                    "feature": self.feature_names[idx],
                    "mean_abs_shap": round(float(mean_abs[idx]), 4),
                    "direction": "positive" if mean_signed[idx] > 0 else "negative",
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Force plot
    # ------------------------------------------------------------------
    def plot_force(
        self, X_single: Union[np.ndarray, pd.DataFrame], save_path: Union[str, Path]
    ) -> None:
        """SHAP force plot for a single prediction."""
        if isinstance(X_single, pd.DataFrame):
            x_arr = X_single.values
        else:
            x_arr = np.asarray(X_single)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        if self.model_type == "logistic_regression" and self.explainer is None:
            self.explainer = shap.LinearExplainer(self.model, x_arr)

        raw = self.explainer.shap_values(x_arr)
        if isinstance(raw, list):
            sv = np.asarray(raw[1] if len(raw) > 1 else raw[0]).flatten()
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                sv = raw[0, :, 1].flatten() if raw.shape[2] == 2 else raw[0, :, 0].flatten()
            elif raw.ndim == 2:
                sv = raw[0, :].flatten()
            else:
                sv = raw.flatten()
        else:
            sv = np.asarray(raw).flatten()

        base_val = self._extract_base_value()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shap.force_plot(
                base_val,
                sv,
                x_arr.flatten(),
                feature_names=self.feature_names,
                matplotlib=True,
                show=False,
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            # Fallback for environments where JS force plot fails
            self._fallback_waterfall(sv, x_arr.flatten(), base_val, save_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _ensure_computed(self) -> None:
        if self.shap_values is None:
            raise RuntimeError(
                "SHAP values not computed yet. Call compute_shap_values() first."
            )

    def _fallback_waterfall(
        self,
        sv: np.ndarray,
        data: np.ndarray,
        base_val: float,
        save_path: Path,
    ) -> None:
        """Simple horizontal bar chart as fallback for waterfall / force."""
        order = np.argsort(np.abs(sv))[-15:]
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#ff4444" if sv[int(i)] > 0 else "#4444ff" for i in order]
        vals = [sv[int(i)] for i in order]
        names = [self.feature_names[int(i)] for i in order]
        ax.barh(range(len(order)), vals, color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Feature contributions (base value = {base_val:.4f})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
