"""Dataset downloading, loading, and splitting utilities.

Handles the UCI Default of Credit Card Clients dataset:
  - Download the .xls from UCI and convert to CSV.
  - Load the CSV into a clean DataFrame.
  - Provide a stratified train/test split.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Tuple
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_config, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(force: bool = False) -> Path:
    """Download the UCI credit-default .xls file and convert to CSV.

    The file is cached: a second call is a no-op unless *force* is True.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    cfg = get_config()
    raw_dir = cfg.data.raw_dir
    csv_path = cfg.data.raw_dataset_path
    xls_path = raw_dir / "credit_default.xls"

    if csv_path.exists() and not force:
        logger.info("Dataset already cached at %s", csv_path)
        return csv_path

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download the xls file with User-Agent header
    url = cfg.data.dataset_url
    logger.info("Downloading dataset from %s …", url)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urlopen(req) as response, open(xls_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    logger.info("Downloaded to %s", xls_path)

    # Convert xls → CSV
    # The UCI file has a header row at row 0 (column names X1..X24) and row 1 has feature names
    try:
        df = pd.read_excel(xls_path, header=1, engine="xlrd")
    except Exception:
        df = pd.read_excel(xls_path, header=1)

    df.to_csv(csv_path, index=False)
    logger.info("Converted to CSV at %s (%d rows, %d cols)", csv_path, *df.shape)

    # Clean up xls
    try:
        xls_path.unlink()
    except OSError:
        pass

    return csv_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the credit-default CSV into a DataFrame.

    - Drops the ``ID`` column if present.
    - Renames ``default.payment.next.month`` → ``default``.

    Parameters
    ----------
    path : Path or str, optional
        Explicit path to CSV. Defaults to ``data/raw/credit_default.csv``.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = get_config().data.raw_dataset_path

    path = Path(path)
    if not path.exists():
        logger.info("CSV not found at %s — downloading first.", path)
        download_dataset()

    df = pd.read_csv(path)
    logger.info("Loaded dataset: %d rows × %d cols", *df.shape)

    # Drop ID column
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Rename target
    target_candidates = [
        "default.payment.next.month",
        "default payment next month",
        "default_payment_next_month",
        "DEFAULT_PAYMENT_NEXT_MONTH",
    ]
    for col in target_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "default"})
            break

    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def get_train_test_split(
    df: pd.DataFrame | None = None,
    target_col: str = "default",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a stratified train/test split.

    Parameters
    ----------
    df : DataFrame, optional
        Dataset to split.  Loaded from disk when *None*.
    target_col : str
        Name of the target column.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    cfg = get_config()

    if df is None:
        df = load_dataset()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.model.test_size,
        random_state=cfg.model.random_state,
        stratify=y,
    )

    logger.info(
        "Split: train=%d, test=%d (%.0f%% test, stratified)",
        len(X_train),
        len(X_test),
        cfg.model.test_size * 100,
    )
    return X_train, X_test, y_train, y_test
