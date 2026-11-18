"""
Customer Risk Segmentation for FinRisk AI.

Provides clustering (K-Means), optimal cluster detection, customer segment
profiling, and risk-tier attribution based on credit risk characteristics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class RiskSegmenter:
    """Segments credit customers into distinct risk profiles using K-Means clustering.

    Includes automatic k-selection, segment risk profiling, visual projections
    via PCA, and feature differentiation for high-risk cohorts.
    """

    DEFAULT_RISK_LABELS_4 = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Parameters
        ----------
        n_clusters : int, default=4
            Number of customer clusters to create.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.segment_label_map: Dict[int, str] = {}
        self.profiles_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fitting and Prediction
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "RiskSegmenter":
        """Scale data and fit K-Means clustering.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : optional ground-truth binary targets (0/1) for default rate profiling
        feature_names : optional list of feature names if X is np.ndarray

        Returns
        -------
        self : fitted instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            if feature_names is not None:
                self.feature_names = list(feature_names)
            elif self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        X_scaled = self.scaler.fit_transform(X_arr)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(X_scaled)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_

        # Cache profiles computed on training data
        self.profiles_ = self.get_segment_profiles(X, y=y, feature_names=feature_names)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict cluster assignment for new samples after scaling.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)

        Returns
        -------
        cluster_labels : np.ndarray of shape (n_samples,)
        """
        if self.kmeans is None:
            raise RuntimeError("RiskSegmenter is not fitted yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        X_scaled = self.scaler.transform(X_arr)
        return self.kmeans.predict(X_scaled)

    # ------------------------------------------------------------------
    # Optimal Cluster Selection
    # ------------------------------------------------------------------
    def find_optimal_clusters(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        max_k: int = 10,
        save_dir: Union[str, Path] = "outputs/segmentation_plots",
    ) -> int:
        """Evaluate inertia (elbow) and silhouette scores across k in [2, max_k].

        Plots elbow curve and silhouette scores to save_dir, and returns optimal k.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        max_k : int, default=10
            Upper bound of clusters to evaluate.
        save_dir : directory to save diagnostic plots.

        Returns
        -------
        optimal_k : int with the highest silhouette score.
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Subsample if dataset is very large to speed up silhouette computation
        if X_arr.shape[0] > 5000:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X_arr.shape[0], size=5000, replace=False)
            X_eval = X_arr[idx]
        else:
            X_eval = X_arr

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_eval)

        k_values = list(range(2, max_k + 1))
        inertias: List[float] = []
        silhouettes: List[float] = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_preds = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_score = silhouette_score(X_scaled, cluster_preds)
            silhouettes.append(sil_score)

        save_path_dir = Path(save_dir)
        save_path_dir.mkdir(parents=True, exist_ok=True)
        plot_file = save_path_dir / "optimal_clusters_elbow_silhouette.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow curve
        ax1.plot(k_values, inertias, "bo-", linewidth=2, markersize=7)
        ax1.set_title("Elbow Method (Inertia vs. k)", fontsize=12)
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Silhouette score curve
        ax2.plot(k_values, silhouettes, "ro-", linewidth=2, markersize=7)
        ax2.set_title("Silhouette Scores vs. k", fontsize=12)
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.grid(True, linestyle="--", alpha=0.6)

        best_idx = int(np.argmax(silhouettes))
        optimal_k = k_values[best_idx]
        ax2.axvline(x=optimal_k, color="green", linestyle=":", label=f"Optimal k={optimal_k}")
        ax2.legend()

        plt.suptitle("Cluster Optimization Analysis", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        return optimal_k

    # ------------------------------------------------------------------
    # Segment Profiling
    # ------------------------------------------------------------------
    def get_segment_profiles(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate summary profile DataFrame for each discovered segment.

        Parameters
        ----------
        X : raw/unscaled feature matrix, or None to use cached training profiles
        y : optional default targets (0 or 1)
        feature_names : optional list of feature names

        Returns
        -------
        profiles_df : pd.DataFrame containing count, percentage, default rate,
                      risk label, and mean values for each feature per cluster.
        """
        if self.kmeans is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if X is None:
            if self.profiles_ is not None:
                return self.profiles_
            # Reconstruct basic profile from cluster centers if X wasn't provided
            centers_orig = self.scaler.inverse_transform(self.cluster_centers_)
            cols = (
                self.feature_names
                if self.feature_names is not None
                else [f"feature_{i}" for i in range(centers_orig.shape[1])]
            )
            rows = []
            for c in range(self.n_clusters):
                row: Dict[str, Any] = {
                    "cluster": c,
                    "count": int((self.labels_ == c).sum()) if self.labels_ is not None else 0,
                    "percentage": round(
                        float((self.labels_ == c).mean() * 100) if self.labels_ is not None else 0.0,
                        2,
                    ),
                    "risk_label": self.segment_label_map.get(c, f"Segment {c}"),
                }
                for j, col in enumerate(cols):
                    row[f"mean_{col}"] = float(centers_orig[c, j])
                rows.append(row)
            return pd.DataFrame(rows)

        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            df = X.copy()
        else:
            cols = (
                feature_names
                if feature_names is not None
                else (self.feature_names or [f"f_{i}" for i in range(X.shape[1])])
            )
            df = pd.DataFrame(X, columns=cols)

        df["cluster"] = self.labels_
        total_samples = len(df)

        if y is not None:
            if isinstance(y, pd.Series):
                y_arr = y.values
            else:
                y_arr = np.asarray(y)
            df["target"] = y_arr

        # Aggregate cluster means and counts
        cluster_rows = []
        for c in range(self.n_clusters):
            c_mask = df["cluster"] == c
            c_df = df[c_mask]
            count = len(c_df)
            pct = (count / total_samples) * 100.0 if total_samples > 0 else 0.0

            row: Dict[str, Any] = {
                "cluster": c,
                "count": count,
                "percentage": round(pct, 2),
            }

            if y is not None:
                c_default_rate = c_df["target"].mean() if count > 0 else 0.0
                row["default_rate"] = float(c_default_rate)

            # Feature means
            for col in cols:
                row[f"mean_{col}"] = float(c_df[col].mean()) if count > 0 else 0.0

            cluster_rows.append(row)

        profiles_df = pd.DataFrame(cluster_rows)

        # Label segments based on default_rate if provided
        if y is not None:
            # Sort cluster indices by default_rate ascending
            sorted_clusters = (
                profiles_df.sort_values(by="default_rate")["cluster"].tolist()
            )

            # Assign labels
            label_map: Dict[int, str] = {}
            if self.n_clusters == 4:
                for idx, cid in enumerate(sorted_clusters):
                    label_map[cid] = self.DEFAULT_RISK_LABELS_4[idx]
            else:
                for idx, cid in enumerate(sorted_clusters):
                    rank_pct = idx / max(1, self.n_clusters - 1)
                    if rank_pct <= 0.25:
                        label_map[cid] = "Low Risk"
                    elif rank_pct <= 0.50:
                        label_map[cid] = "Medium Risk"
                    elif rank_pct <= 0.75:
                        label_map[cid] = "High Risk"
                    else:
                        label_map[cid] = "Very High Risk"

            self.segment_label_map = label_map
            profiles_df["risk_label"] = profiles_df["cluster"].map(label_map)
        else:
            self.segment_label_map = {c: f"Segment {c}" for c in range(self.n_clusters)}
            profiles_df["risk_label"] = profiles_df["cluster"].map(self.segment_label_map)

        self.profiles_ = profiles_df
        return profiles_df

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def plot_segments(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        save_path: Union[str, Path],
        method: str = "pca",
    ) -> None:
        """Project features onto 2D space (PCA) and plot customer clusters.

        Parameters
        ----------
        X : feature matrix
        save_path : destination path for the saved scatter plot
        method : dimensionality reduction method (defaults to 'pca')
        """
        if self.kmeans is None or self.labels_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        X_scaled = self.scaler.transform(X_arr)

        if method.lower() == "pca":
            pca = PCA(n_components=2, random_state=self.random_state)
            coords_2d = pca.fit_transform(X_scaled)
            centers_2d = pca.transform(self.cluster_centers_)
            var_exp = pca.explained_variance_ratio_ * 100
            x_label = f"PCA Component 1 ({var_exp[0]:.1f}% var)"
            y_label = f"PCA Component 2 ({var_exp[1]:.1f}% var)"
        else:
            # Fallback to first two dimensions
            coords_2d = X_scaled[:, :2]
            centers_2d = self.cluster_centers_[:, :2]
            x_label = "Feature 1 (scaled)"
            y_label = "Feature 2 (scaled)"

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.get_cmap("tab10", self.n_clusters)

        for c in range(self.n_clusters):
            mask = self.labels_ == c
            label_text = self.segment_label_map.get(c, f"Cluster {c}")
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                s=20,
                alpha=0.45,
                color=colors(c),
                label=f"{label_text} (n={mask.sum()})",
            )

        # Plot and annotate centroids
        for c in range(self.n_clusters):
            cx, cy = centers_2d[c, 0], centers_2d[c, 1]
            ax.scatter(
                cx,
                cy,
                marker="X",
                s=200,
                color="black",
                edgecolors="white",
                linewidth=1.5,
                zorder=5,
            )
            label_text = self.segment_label_map.get(c, f"Cluster {c}")
            ax.annotate(
                f" {label_text}",
                (cx, cy),
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
            )

        ax.set_title("Customer Risk Segmentation Map (2D PCA)", fontsize=14, pad=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_segment_profiles(
        self,
        profiles_df: pd.DataFrame,
        save_path: Union[str, Path],
        top_n_features: int = 8,
    ) -> None:
        """Plot standardized feature comparisons across risk segments.

        Parameters
        ----------
        profiles_df : DataFrame returned by get_segment_profiles()
        save_path : output file destination
        top_n_features : max features with highest variance to highlight
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract mean columns
        mean_cols = [c for c in profiles_df.columns if c.startswith("mean_")]
        if not mean_cols:
            raise ValueError("No 'mean_*' columns found in profiles_df.")

        # Normalize features across clusters (z-score like standardisation) for visual comparison
        sub_df = profiles_df[mean_cols].copy()
        clean_names = [c.replace("mean_", "") for c in mean_cols]
        sub_df.columns = clean_names

        # Pick top_n_features with highest cross-segment variance
        variances = sub_df.var(axis=0)
        top_feats = variances.nlargest(top_n_features).index.tolist()
        sub_top = sub_df[top_feats]

        # Min-max scale per feature across clusters so they can be shown on the same scale
        normalized = (sub_top - sub_top.min(axis=0)) / (
            sub_top.max(axis=0) - sub_top.min(axis=0) + 1e-9
        )

        labels = (
            profiles_df["risk_label"]
            if "risk_label" in profiles_df.columns
            else [f"Cluster {c}" for c in profiles_df["cluster"]]
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top_feats))
        width = 0.8 / len(profiles_df)

        for i, (idx, row) in enumerate(normalized.iterrows()):
            ax.bar(
                x + i * width - (len(profiles_df) - 1) * width / 2,
                row.values,
                width,
                label=f"{labels[idx]}",
            )

        ax.set_title("Relative Feature Profile Across Risk Segments", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel("Normalized Feature Level (0 = Min, 1 = Max)")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Risk Segment", loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # High-Risk Cluster Analysis
    # ------------------------------------------------------------------
    def get_high_risk_characteristics(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Identify high-risk segment and its most distinguishing characteristics.

        Parameters
        ----------
        X : raw/unscaled feature matrix
        y : binary default labels (0 or 1)
        feature_names : optional list of feature names
        top_n : number of top distinguishing features to extract

        Returns
        -------
        result : Dict with keys:
            cluster_id : int
            size : int
            default_rate : float
            distinguishing_features : List[Dict] with feature name, cluster mean,
                                      overall mean, and z-score difference.
        """
        if self.labels_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            df = X.copy()
        else:
            cols = (
                feature_names
                if feature_names is not None
                else (self.feature_names or [f"f_{i}" for i in range(X.shape[1])])
            )
            df = pd.DataFrame(X, columns=cols)

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        df["cluster"] = self.labels_
        df["target"] = y_arr

        # Find cluster with highest default rate
        default_rates = df.groupby("cluster")["target"].mean()
        high_risk_cluster = int(default_rates.idxmax())
        high_risk_rate = float(default_rates.max())

        high_risk_df = df[df["cluster"] == high_risk_cluster]
        cluster_size = len(high_risk_df)

        # Distinguishing features based on standardized difference from overall mean
        overall_means = df[cols].mean()
        overall_stds = df[cols].std().replace(0, 1e-9)
        cluster_means = high_risk_df[cols].mean()

        # Z-difference: (cluster_mean - overall_mean) / overall_std
        z_diffs = (cluster_means - overall_means) / overall_stds
        top_diff_indices = z_diffs.abs().nlargest(top_n).index

        distinguishing: List[Dict[str, Any]] = []
        for feat in top_diff_indices:
            distinguishing.append(
                {
                    "feature": feat,
                    "cluster_mean": round(float(cluster_means[feat]), 4),
                    "overall_mean": round(float(overall_means[feat]), 4),
                    "z_score_diff": round(float(z_diffs[feat]), 4),
                    "relative_change_pct": round(
                        float(
                            (cluster_means[feat] - overall_means[feat])
                            / (abs(overall_means[feat]) + 1e-9)
                            * 100
                        ),
                        2,
                    ),
                }
            )

        return {
            "cluster_id": high_risk_cluster,
            "size": cluster_size,
            "default_rate": round(high_risk_rate, 4),
            "distinguishing_features": distinguishing,
        }
