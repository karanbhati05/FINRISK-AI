"""Feature engineering for the credit-default dataset.

``FeatureEngineer.transform(df)`` adds derived columns that capture
utilisation behaviour, payment patterns, and demographic bins.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column groups
PAY_COLS = [f"PAY_{i}" for i in range(0, 7)]       # PAY_0 … PAY_6
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]  # BILL_AMT1 … BILL_AMT6
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]  # PAY_AMT1 … PAY_AMT6


class FeatureEngineer:
    """Stateless feature-engineering transformer.

    Usage::

        fe = FeatureEngineer()
        df = fe.transform(df)
    """

    # Names of all engineered features (for reference by downstream code)
    ENGINEERED_FEATURES: List[str] = [
        "utilization_ratio",
        "avg_payment_delay",
        "max_payment_delay",
        "payment_to_bill_ratio",
        "total_bill",
        "total_payment",
        "payment_consistency",
        "age_group",
        "credit_limit_category",
        "has_delayed_payment",
        "num_delayed_months",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features and return a new DataFrame.

        The original columns are preserved.
        """
        df = df.copy()

        # Utilisation ratio
        denom_util = df["LIMIT_BAL"] + 1
        denom_util = denom_util.replace(0, 1)
        df["utilization_ratio"] = (df["BILL_AMT1"] / denom_util).replace([np.inf, -np.inf], np.nan)

        # Payment-delay statistics
        pay_cols = [c for c in ["PAY_0", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in df.columns]
        if not pay_cols:
            pay_cols = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
        pay_delay = df[pay_cols]
        df["avg_payment_delay"] = pay_delay.mean(axis=1)
        df["max_payment_delay"] = pay_delay.max(axis=1)

        # Payment-to-bill ratio (most recent month)
        denom_bill = df["BILL_AMT1"] + 1
        denom_bill = denom_bill.replace(0, 1)
        df["payment_to_bill_ratio"] = (df["PAY_AMT1"] / denom_bill).replace([np.inf, -np.inf], np.nan)

        # Aggregate bill & payment amounts
        df["total_bill"] = df[BILL_COLS].sum(axis=1)
        df["total_payment"] = df[PAY_AMT_COLS].sum(axis=1)

        # Payment consistency (lower std → more consistent)
        df["payment_consistency"] = df[PAY_AMT_COLS].std(axis=1)

        # Age groups
        df["age_group"] = pd.cut(
            df["AGE"],
            bins=[0, 30, 50, 100],
            labels=["young", "middle", "senior"],
        ).astype(str)

        # Credit-limit categories
        df["credit_limit_category"] = pd.cut(
            df["LIMIT_BAL"],
            bins=[0, 100_000, 250_000, 500_000, float("inf")],
            labels=["low", "medium", "high", "very_high"],
        ).astype(str)

        # Delayed-payment flags
        df["has_delayed_payment"] = (pay_delay > 0).any(axis=1).astype(int)
        df["num_delayed_months"] = (pay_delay > 0).sum(axis=1)

        logger.info(
            "Engineered %d new features → %d total columns",
            len(self.ENGINEERED_FEATURES),
            df.shape[1],
        )
        return df
