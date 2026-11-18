"""Data preprocessing pipeline for FinRisk AI.

Provides a scikit-learn-compatible ``DataPreprocessor`` with fit/transform,
handling:
  - Cleaning invalid EDUCATION and MARRIAGE codes.
  - Imputing missing values (median for numeric, mode for categorical).
  - Scaling numeric features with StandardScaler.
  - Serialisation via joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import get_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Columns that are encoded integers (categorical semantics, no scaling)
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Pay-status columns
PAY_COLS = [f"PAY_{i}" for i in range(0, 7)]  # PAY_0 … PAY_6

# Bill / payment amount columns
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


class DataPreprocessor:
    """Fit / transform preprocessor for the credit-default dataset.

    Usage::

        prep = DataPreprocessor()
        X_train_t = prep.fit_transform(X_train)
        X_test_t  = prep.transform(X_test)
        prep.save("models/preprocessor.joblib")
    """

    def __init__(self) -> None:
        self._pipeline: Optional[ColumnTransformer] = None
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._str_cat_cols: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clean_education(df: pd.DataFrame) -> pd.DataFrame:
        """Map EDUCATION values 0, 4, 5, 6 → 4 (other)."""
        df = df.copy()
        if "EDUCATION" in df.columns:
            df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        return df

    @staticmethod
    def clean_marriage(df: pd.DataFrame) -> pd.DataFrame:
        """Map MARRIAGE value 0 → 3 (other)."""
        df = df.copy()
        if "MARRIAGE" in df.columns:
            df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
        return df

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps."""
        df = self.clean_education(df)
        df = self.clean_marriage(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> None:
        """Determine numeric vs categorical columns from the DataFrame."""
        self._categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        self._numeric_cols = [
            c
            for c in df.columns
            if c not in self._categorical_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        self._str_cat_cols = [
            c
            for c in df.columns
            if c not in self._categorical_cols
            and c not in self._numeric_cols
        ]

    def _build_pipeline(self) -> ColumnTransformer:
        """Build the ColumnTransformer pipeline."""
        transformers = []

        if self._numeric_cols:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_pipeline, self._numeric_cols))

        if self._categorical_cols:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            )
            transformers.append(("cat", categorical_pipeline, self._categorical_cols))

        if self._str_cat_cols:
            str_cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformers.append(("str_cat", str_cat_pipeline, self._str_cat_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """Fit the preprocessor on training data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (*without* the target column).
        """
        X = self._clean(X)
        self._resolve_columns(X)
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)
        self._fitted = True
        logger.info(
            "Preprocessor fitted — %d numeric, %d categorical, %d str-categorical features",
            len(self._numeric_cols),
            len(self._categorical_cols),
            len(self._str_cat_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using the fitted pipeline.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Transformed feature array.
        """
        if not self._fitted:
            raise RuntimeError("DataPreprocessor has not been fitted yet.")
        X = self._clean(X)
        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return feature names after transformation."""
        if not self._fitted:
            raise RuntimeError("DataPreprocessor has not been fitted yet.")
        return list(self._numeric_cols) + list(self._categorical_cols) + list(self._str_cat_cols)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the preprocessor to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DataPreprocessor":
        """Load a previously saved preprocessor."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DataPreprocessor, got {type(obj).__name__}")
        logger.info("Preprocessor loaded from %s", path)
        return obj
