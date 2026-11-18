"""Configuration loader for FinRisk AI.

Loads config.yaml and provides typed access to all settings via a Config dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    dataset_url: str = ""
    dataset_filename: str = "credit_default.csv"
    target_column: str = "default"

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self.raw_path

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.processed_path

    @property
    def raw_dataset_path(self) -> Path:
        return self.raw_dir / self.dataset_filename


@dataclass
class LogisticRegressionConfig:
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    class_weight: str = "balanced"


@dataclass
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 3.5
    random_state: int = 42
    eval_metric: str = "auc"
    use_label_encoder: bool = False


@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)


@dataclass
class TrainingConfig:
    early_stopping_rounds: int = 20
    validation_fraction: float = 0.1


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "FinRisk AI API"
    version: str = "0.1.0"
    reload: bool = True


@dataclass
class MLflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "finrisk-credit-default"
    artifact_location: str = "mlruns/artifacts"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "Config":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : Path or str, optional
            Path to the YAML config file. Defaults to ``config.yaml``
            in the project root.
        """
        path = Path(path) if path else CONFIG_PATH
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=_build_model_config(raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            api=APIConfig(**raw.get("api", {})),
            mlflow=MLflowConfig(**raw.get("mlflow", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )


def _build_model_config(raw: Dict[str, Any]) -> ModelConfig:
    """Build a ModelConfig, nesting sub-model configs."""
    lr = LogisticRegressionConfig(**raw.pop("logistic_regression", {}))
    rf = RandomForestConfig(**raw.pop("random_forest", {}))
    xgb = XGBoostConfig(**raw.pop("xgboost", {}))
    return ModelConfig(**raw, logistic_regression=lr, random_forest=rf, xgboost=xgb)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------
_config: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Return the global Config singleton, loading from disk on first call."""
    global _config
    if _config is None or reload:
        _config = Config.from_yaml()
    return _config
