"""
FinRisk AI - API route definitions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerData,
    DriftReport,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
)

logger = logging.getLogger("finrisk.api")

router = APIRouter()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "monitoring"

# ---------------------------------------------------------------------------
# Dependency-injection helpers (cached singletons)
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Any] = {}


def _load_artifact(key: str, path: Path, loader=joblib.load):
    """Load an artifact once and cache it."""
    if key not in _model_cache:
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Artifact not found: {path}. Train the model first.",
            )
        _model_cache[key] = loader(path)
        logger.info("Loaded %s from %s", key, path)
    return _model_cache[key]


def get_model():
    return _load_artifact("model", MODELS_DIR / "best_model.joblib")


def get_preprocessor():
    return _load_artifact("preprocessor", MODELS_DIR / "preprocessor.joblib")


def get_feature_engineer():
    return _load_artifact("feature_engineer", MODELS_DIR / "feature_engineer.joblib")


def _get_optional_artifact(key: str, path: Path):
    """Return cached artifact or None when file is missing."""
    if key in _model_cache:
        return _model_cache[key]
    if path.exists():
        _model_cache[key] = joblib.load(path)
        return _model_cache[key]
    return None


def get_explainer():
    return _get_optional_artifact("explainer", MODELS_DIR / "shap_explainer.joblib")


def get_segmenter():
    return _get_optional_artifact("segmenter", MODELS_DIR / "risk_segmenter.joblib")


def get_drift_detector():
    return _get_optional_artifact("drift_detector", MODELS_DIR / "drift_detector.joblib")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_ORDER: List[str] = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

DEFAULT_THRESHOLD = 0.5


def _customer_to_df(customer: CustomerData) -> pd.DataFrame:
    """Convert a single CustomerData to a one-row DataFrame."""
    return pd.DataFrame([customer.model_dump()])[FEATURE_ORDER]


def _classify_risk(prob: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    if prob >= threshold:
        return "High Risk"
    elif prob >= threshold * 0.6:
        return "Medium Risk"
    return "Low Risk"


def _run_pipeline(
    df: pd.DataFrame,
    model,
    preprocessor=None,
    feature_engineer=None,
) -> np.ndarray:
    """Apply feature engineering + preprocessing + predict_proba."""
    if feature_engineer is not None:
        try:
            df = feature_engineer.transform(df)
        except Exception:
            logger.warning("Feature engineering failed; using raw features")
    if preprocessor is not None:
        try:
            df = preprocessor.transform(df)
        except Exception:
            logger.warning("Preprocessing failed; using current features")
    probas = model.predict_proba(df)
    # predict_proba may return (n, 2) or (n,)
    if probas.ndim == 2:
        return probas[:, 1]
    return probas


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(
    customer: CustomerData,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
    feature_engineer=Depends(get_feature_engineer),
):
    """Predict default probability for a single customer."""
    df = _customer_to_df(customer)
    proba = _run_pipeline(df, model, preprocessor, feature_engineer)[0]

    segmenter = get_segmenter()
    segment = None
    if segmenter is not None:
        try:
            segment = str(segmenter.predict(df)[0])
        except Exception:
            pass

    return PredictionResponse(
        default_probability=round(float(proba), 4),
        risk_label=_classify_risk(float(proba)),
        risk_segment=segment,
        threshold_used=DEFAULT_THRESHOLD,
        model_used=type(model).__name__,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    request: BatchPredictionRequest,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
    feature_engineer=Depends(get_feature_engineer),
):
    """Predict default probability for a batch of customers."""
    rows = [c.model_dump() for c in request.customers]
    df = pd.DataFrame(rows)[FEATURE_ORDER]
    probas = _run_pipeline(df, model, preprocessor, feature_engineer)

    predictions = []
    for i, prob in enumerate(probas):
        predictions.append(
            PredictionResponse(
                customer_id=str(i),
                default_probability=round(float(prob), 4),
                risk_label=_classify_risk(float(prob)),
                threshold_used=DEFAULT_THRESHOLD,
                model_used=type(model).__name__,
            )
        )

    high_risk = sum(1 for p in predictions if p.risk_label == "High Risk")
    return BatchPredictionResponse(
        predictions=predictions,
        summary={
            "total": len(predictions),
            "high_risk_count": high_risk,
            "high_risk_percentage": round(high_risk / len(predictions) * 100, 2),
        },
    )


@router.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_prediction(
    customer: CustomerData,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
    feature_engineer=Depends(get_feature_engineer),
):
    """Return prediction with SHAP-based explanation."""
    df = _customer_to_df(customer)
    proba = _run_pipeline(df, model, preprocessor, feature_engineer)[0]

    explainer = get_explainer()
    if explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available. Train with SHAP enabled.")

    try:
        explanation = explainer.explain_prediction(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    # Support both explain_prediction formats (structured dicts or shap_values map)
    top_risk = explanation.get("top_risk_factors")
    top_protective = explanation.get("top_protective_factors")
    explanation_text = explanation.get("risk_explanation")

    if top_risk is None or top_protective is None:
        shap_values = explanation.get("shap_values", {})
        risk_factors = []
        protective_factors = []
        for feature, sv in sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True):
            entry = {
                "feature": feature,
                "value": float(df[feature].iloc[0]) if feature in df.columns else None,
                "shap_value": round(float(sv), 4),
                "direction": "positive" if sv > 0 else "negative",
            }
            if sv > 0:
                risk_factors.append(entry)
            else:
                protective_factors.append(entry)
        top_risk = risk_factors[:5]
        top_protective = protective_factors[:5]

    if not explanation_text:
        if top_risk:
            factors_str = ", ".join(f["feature"] for f in top_risk[:3])
            explanation_text = (
                f"This customer has a {proba:.1%} probability of default. "
                f"The top risk drivers are: {factors_str}."
            )
        else:
            explanation_text = f"This customer has a {proba:.1%} probability of default with no strong risk drivers."

    return ExplanationResponse(
        prediction=PredictionResponse(
            default_probability=round(float(proba), 4),
            risk_label=_classify_risk(float(proba)),
            threshold_used=DEFAULT_THRESHOLD,
            model_used=type(model).__name__,
        ),
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
        risk_explanation=explanation_text,
    )


@router.get("/risk-factors", tags=["Explainability"])
def global_risk_factors():
    """Return global SHAP-based top risk factors."""
    explainer = get_explainer()
    if explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available.")
    try:
        df = explainer.get_top_risk_factors(n=10)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not compute risk factors: {exc}")


@router.get("/segments", tags=["Segmentation"])
def segment_profiles():
    """Return risk-segment profiles."""
    segmenter = get_segmenter()
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Risk segmenter not available.")
    try:
        if hasattr(segmenter, "profiles_") and segmenter.profiles_ is not None:
            df = segmenter.profiles_
        else:
            try:
                df = segmenter.get_segment_profiles()
            except TypeError:
                X_ref = getattr(segmenter, "X_", getattr(segmenter, "X_sample_", None))
                if X_ref is not None:
                    df = segmenter.get_segment_profiles(X_ref)
                else:
                    raise
        if isinstance(df, pd.DataFrame):
            return df.to_dict(orient="records")
        return df
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not retrieve segments: {exc}")


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Service health check."""
    model = _model_cache.get("model")
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else "none",
        version="0.1.0",
    )


@router.get("/drift-report", response_model=DriftReport, tags=["Monitoring"])
def latest_drift_report():
    """Return the most recent drift report from disk."""
    import json

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(OUTPUTS_DIR.glob("drift_report_*.json"), reverse=True)
    if not reports:
        raise HTTPException(status_code=404, detail="No drift report available yet.")
    with open(reports[0]) as f:
        data = json.load(f)
    return DriftReport(**data)


@router.post("/monitor/check-drift", response_model=DriftReport, tags=["Monitoring"])
def trigger_drift_check(customers: List[CustomerData]):
    """Run drift detection on a new batch of data."""
    detector = get_drift_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialised.")

    rows = [c.model_dump() for c in customers]
    new_data = pd.DataFrame(rows)[FEATURE_ORDER]

    try:
        result = detector.check_drift(new_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {exc}")

    return DriftReport(
        is_drifted=result["is_drifted"],
        drift_score=round(result["drift_score"], 4),
        drifted_features=result["drifted_features"],
        report_timestamp=datetime.now(timezone.utc).isoformat(),
    )
