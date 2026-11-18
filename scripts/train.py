#!/usr/bin/env python
"""
FinRisk AI — Training Orchestrator

End-to-end pipeline that:
  1. Loads the UCI Credit Default dataset (with fallback)
  2. Engineers features (with fallback)
  3. Preprocesses (with fallback to StandardScaler)
  4. Trains 4 models (LR+SMOTE, RF+SMOTE, XGB+SMOTE, XGB+class_weight)
  5. Evaluates, compares, tunes thresholds, and generates plots
  6. Saves the best model to models/best_model.joblib
  7. Tracks everything in MLflow (experiment: finrisk-credit-default)

Usage:
    python scripts/train.py
"""

import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so "src.*" imports work when running
# the script directly (e.g. `python scripts/train.py`).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("finrisk.train")

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
try:
    import mlflow

    mlflow.set_experiment("finrisk-credit-default")
    logger.info("MLflow experiment set to 'finrisk-credit-default'")
except Exception as exc:
    logger.warning("MLflow setup failed (%s) — training will proceed without tracking.", exc)
    mlflow = None


# ===================================================================
# 1. LOAD DATA
# ===================================================================
def load_data() -> pd.DataFrame:
    """Load the credit default dataset, trying src.data.loader first."""
    try:
        from src.data.loader import load_dataset

        df = load_dataset()
        logger.info("Data loaded via src.data.loader (%d rows)", len(df))
        return df
    except Exception as exc:
        logger.warning("src.data.loader unavailable (%s), using fallback.", exc)

    # Fallback: load directly
    raw_path = PROJECT_ROOT / "data" / "raw" / "credit_default.csv"
    if not raw_path.exists():
        # Try the xls file
        xls_path = PROJECT_ROOT / "data" / "raw" / "credit_default.xls"
        if xls_path.exists():
            df = pd.read_excel(xls_path, header=1)
        else:
            raise FileNotFoundError(
                f"No data file found at {raw_path} or {xls_path}. "
                "Run the data loader first or place the dataset manually."
            )
    else:
        df = pd.read_csv(raw_path)

    # Clean up column names
    if "default.payment.next.month" in df.columns:
        df = df.rename(columns={"default.payment.next.month": "default"})
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    logger.info("Data loaded via fallback (%d rows, %d cols)", *df.shape)
    return df


# ===================================================================
# 2. FEATURE ENGINEERING
# ===================================================================
def engineer_features(df: pd.DataFrame):
    """Apply feature engineering — try the shared module, else skip."""
    fe = None
    try:
        from src.features.engineer import FeatureEngineer

        fe = FeatureEngineer()
        df = fe.transform(df)
        logger.info("Feature engineering applied via src.features.engineer")
    except Exception as exc:
        logger.warning(
            "src.features.engineer unavailable (%s), skipping feature engineering.", exc
        )
    return df, fe


# ===================================================================
# 3. PREPROCESS
# ===================================================================
def preprocess(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple:
    """Scale features — try the shared preprocessor, else StandardScaler."""
    try:
        from src.data.preprocessor import DataPreprocessor

        prep = DataPreprocessor()
        X_train_proc = prep.fit_transform(X_train)
        X_test_proc = prep.transform(X_test)
        logger.info("Preprocessing applied via src.data.preprocessor")
        return X_train_proc, X_test_proc, prep
    except Exception as exc:
        logger.warning(
            "src.data.preprocessor unavailable (%s), falling back to StandardScaler.",
            exc,
        )
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
        return X_train_proc, X_test_proc, scaler


# ===================================================================
# 4. SPLIT
# ===================================================================
def split_data(df: pd.DataFrame):
    """Stratified train/test split — try shared loader, else do it locally."""
    try:
        from src.data.loader import get_train_test_split

        return get_train_test_split(df, test_size=0.2, random_state=42)
    except Exception:
        pass

    target = "default"
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ===================================================================
# MAIN
# ===================================================================
def main():
    from src.models.trainer import ModelTrainer
    from src.models.evaluator import ModelEvaluator

    separator = "=" * 60

    # --- Load & prepare -------------------------------------------------
    print(f"\n{separator}")
    print("  FinRisk AI — Training Pipeline")
    print(separator)

    df = load_data()
    df, fe = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_proc, X_test_proc, prep = preprocess(X_train, X_test)

    logger.info(
        "Split: train=%d, test=%d  |  Default rate: %.1f%%",
        len(y_train),
        len(y_test),
        100 * y_train.mean(),
    )

    # --- Define model configurations ------------------------------------
    model_configs = {
        "LR_SMOTE": ("logistic_regression", "smote"),
        "RF_SMOTE": ("random_forest", "smote"),
        "XGB_SMOTE": ("xgboost", "smote"),
        "XGB_ClassWeight": ("xgboost", "class_weight"),
    }

    # --- Wrap in a top-level MLflow run ----------------------------------
    run_ctx = None
    if mlflow is not None:
        try:
            run_ctx = mlflow.start_run(run_name="full-training-pipeline")
            run_ctx.__enter__()
        except Exception:
            run_ctx = None

    try:
        # --- Train models ---------------------------------------------------
        trained_models: dict[str, ModelTrainer] = {}
        print(f"\n{'─' * 60}")
        print("  Training models …")
        print(f"{'─' * 60}")

        for name, (mtype, imbalance) in model_configs.items():
            print(f"\n  ▸ {name}  ({mtype}, imbalance={imbalance})")
            trainer = ModelTrainer(
                model_type=mtype,
                handle_imbalance=imbalance,
                random_state=42,
            )
            trainer.train(X_train_proc, y_train)
            trained_models[name] = trainer
            print(f"    ✓ trained in {trainer.training_time:.2f}s")

        # --- Evaluate -------------------------------------------------------
        print(f"\n{'─' * 60}")
        print("  Evaluating models …")
        print(f"{'─' * 60}")

        evaluator = ModelEvaluator(trained_models)
        results = evaluator.evaluate(X_test_proc, y_test)

        comparison = evaluator.compare_models(X_test_proc, y_test)
        print("\n  Model Comparison:")
        print(comparison.to_string())

        # --- Plots ----------------------------------------------------------
        print(f"\n{'─' * 60}")
        print("  Generating plots …")
        print(f"{'─' * 60}")

        evaluator.plot_roc_curves(X_test_proc, y_test)
        evaluator.plot_precision_recall_curves(X_test_proc, y_test)
        evaluator.plot_confusion_matrices(X_test_proc, y_test)

        # --- Threshold tuning -----------------------------------------------
        print(f"\n{'─' * 60}")
        print("  Tuning classification thresholds …")
        print(f"{'─' * 60}")

        for name in trained_models:
            opt_t = evaluator.tune_threshold(name, X_test_proc, y_test, metric="f1")
            print(f"    {name}: optimal threshold (F1) = {opt_t:.2f}")

        # --- Best model -----------------------------------------------------
        best_name = evaluator.get_best_model(metric="roc_auc")
        best_model = trained_models[best_name]

        save_dir = PROJECT_ROOT / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_model.save_model(str(save_dir / "best_model.joblib"))

        import joblib
        if prep is not None:
            prep_path = save_dir / "preprocessor.joblib"
            if hasattr(prep, "save"):
                prep.save(str(prep_path))
            else:
                joblib.dump(prep, prep_path)
            logger.info("Preprocessor saved to %s", prep_path)

        if fe is not None:
            fe_path = save_dir / "feature_engineer.joblib"
            joblib.dump(fe, fe_path)
            logger.info("Feature engineer saved to %s", fe_path)

        # --- Summary --------------------------------------------------------
        print(f"\n{separator}")
        print("  RESULTS SUMMARY")
        print(separator)
        print(f"  Best model (ROC-AUC) : {best_name}")
        print(f"  ROC-AUC              : {results[best_name]['roc_auc']:.4f}")
        print(f"  F1 (macro)           : {results[best_name]['f1_macro']:.4f}")
        print(f"  PR-AUC              : {results[best_name]['pr_auc']:.4f}")
        print(f"  Accuracy             : {results[best_name]['accuracy']:.4f}")
        print(f"  Model saved to       : {best_path}")
        if prep is not None:
            print(f"  Preprocessor saved to: {save_dir / 'preprocessor.joblib'}")
        if fe is not None:
            print(f"  Feature engineer saved to: {save_dir / 'feature_engineer.joblib'}")
        print(f"  Plots saved to       : outputs/model_plots/")
        print(separator)

    finally:
        if run_ctx is not None:
            try:
                run_ctx.__exit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    main()
