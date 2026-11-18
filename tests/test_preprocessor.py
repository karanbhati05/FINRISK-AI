"""
Tests for src.data.preprocessor.DataPreprocessor.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _make_sample_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic DataFrame that mimics credit-default features."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "LIMIT_BAL": rng.uniform(10000, 500000, n),
        "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([1, 2, 3, 4], n),
        "MARRIAGE": rng.choice([1, 2, 3], n),
        "AGE": rng.randint(21, 70, n),
        "PAY_0": rng.choice([-1, 0, 1, 2], n),
        "PAY_2": rng.choice([-1, 0, 1, 2], n),
        "PAY_3": rng.choice([-1, 0, 1, 2], n),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""

    def test_import(self):
        """Module can be imported."""
        from src.data.preprocessor import DataPreprocessor  # noqa: F401

    def test_fit_transform_shape(self):
        """fit + transform should return an array with the same number of rows."""
        from src.data.preprocessor import DataPreprocessor

        df = _make_sample_df(80)
        prep = DataPreprocessor()
        prep.fit(df)
        result = prep.transform(df)
        assert result.shape[0] == 80, "Row count must be preserved"
        assert result.shape[1] > 0, "Must produce at least one feature column"

    def test_transform_new_data(self):
        """Transforming unseen data should work once fitted."""
        from src.data.preprocessor import DataPreprocessor

        train = _make_sample_df(80, seed=1)
        test = _make_sample_df(20, seed=2)
        prep = DataPreprocessor()
        prep.fit(train)
        result = prep.transform(test)
        assert result.shape[0] == 20

    def test_handles_missing_values(self):
        """DataPreprocessor should not crash when input has NaN values."""
        from src.data.preprocessor import DataPreprocessor

        df = _make_sample_df(50)
        # Inject some NaNs
        df.loc[0, "LIMIT_BAL"] = np.nan
        df.loc[3, "AGE"] = np.nan
        df.loc[5, "BILL_AMT1"] = np.nan

        prep = DataPreprocessor()
        prep.fit(df)
        result = prep.transform(df)
        assert not np.isnan(result).any(), "Output should have no NaN values"

    def test_output_is_numeric(self):
        """All output values should be numeric (no object dtype)."""
        from src.data.preprocessor import DataPreprocessor

        df = _make_sample_df(60)
        prep = DataPreprocessor()
        prep.fit(df)
        result = prep.transform(df)
        if isinstance(result, pd.DataFrame):
            assert all(np.issubdtype(dt, np.number) for dt in result.dtypes)
        else:
            assert np.issubdtype(result.dtype, np.number)
