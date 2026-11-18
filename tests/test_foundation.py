"""Test foundation modules: config, data loader, preprocessor, and feature engineer."""

import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_foundation")


def test_config():
    logger.info("--- Testing Config ---")
    from src.config import get_config, PROJECT_ROOT, DATA_DIR
    cfg = get_config(reload=True)
    assert cfg is not None
    assert cfg.model.test_size == 0.2
    assert cfg.model.random_state == 42
    assert cfg.model.random_forest.n_estimators == 200
    assert cfg.model.xgboost.n_estimators == 300
    assert cfg.api.port == 8000
    assert cfg.mlflow.experiment_name == "finrisk-credit-default"
    logger.info("Config verified successfully!")


def test_data_loader_and_pipeline():
    logger.info("--- Testing Data Loader & Pipeline ---")
    from src.data.loader import download_dataset, load_dataset, get_train_test_split
    from src.features.engineer import FeatureEngineer
    from src.data.preprocessor import DataPreprocessor

    csv_path = download_dataset()
    assert Path(csv_path).exists(), f"CSV does not exist: {csv_path}"
    logger.info("download_dataset() succeeded: %s", csv_path)

    df = load_dataset()
    logger.info("load_dataset() shape: %s", df.shape)
    assert "ID" not in df.columns, "ID column should be dropped"
    assert "default" in df.columns, "Target column should be 'default'"
    assert df.shape[0] == 30000, f"Expected 30000 rows, got {df.shape[0]}"
    assert df.shape[1] == 24, f"Expected 24 columns, got {df.shape[1]}"

    # Train/test split
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    assert len(X_train) == 24000
    assert len(X_test) == 6000
    assert len(y_train) == 24000
    assert len(y_test) == 6000
    logger.info("Train/test split verified (24000 train, 6000 test).")

    # Feature Engineer
    fe = FeatureEngineer()
    X_train_fe = fe.transform(X_train)
    logger.info("FeatureEngineer transformed shape: %s", X_train_fe.shape)
    for feat in fe.ENGINEERED_FEATURES:
        assert feat in X_train_fe.columns, f"Engineered feature '{feat}' missing"

    # Data Preprocessor
    prep = DataPreprocessor()
    X_train_trans = prep.fit_transform(X_train)
    assert X_train_trans.shape[0] == 24000
    feat_names = prep.get_feature_names()
    assert len(feat_names) == X_train_trans.shape[1]
    logger.info("DataPreprocessor transformed shape: %s with %d feature names", X_train_trans.shape, len(feat_names))

    # Persistence
    save_path = Path("models/test_preprocessor.joblib")
    prep.save(save_path)
    assert save_path.exists()
    loaded_prep = DataPreprocessor.load(save_path)
    X_test_trans = loaded_prep.transform(X_test)
    assert X_test_trans.shape[0] == 6000
    save_path.unlink()
    logger.info("DataPreprocessor save/load verified successfully!")


if __name__ == "__main__":
    test_config()
    test_data_loader_and_pipeline()
    print("\nALL FOUNDATION TESTS PASSED SUCCESSFULLY!\n")
