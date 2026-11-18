"""
FinRisk AI - Pydantic schemas for API request/response models.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Input schema for a single customer's credit data."""

    LIMIT_BAL: float = Field(..., description="Amount of credit (NT dollar)")
    SEX: int = Field(..., ge=1, le=2, description="1=male, 2=female")
    EDUCATION: int = Field(..., ge=1, le=4, description="1=grad school, 2=university, 3=high school, 4=other")
    MARRIAGE: int = Field(..., ge=1, le=3, description="1=married, 2=single, 3=other")
    AGE: int = Field(..., ge=18, le=120, description="Age in years")
    PAY_0: int = Field(..., description="Repayment status in Sep (-1=pay duly, 1=1mo delay, ...)")
    PAY_2: int = Field(..., description="Repayment status in Aug")
    PAY_3: int = Field(..., description="Repayment status in Jul")
    PAY_4: int = Field(..., description="Repayment status in Jun")
    PAY_5: int = Field(..., description="Repayment status in May")
    PAY_6: int = Field(..., description="Repayment status in Apr")
    BILL_AMT1: float = Field(..., description="Bill statement amount in Sep (NT dollar)")
    BILL_AMT2: float = Field(..., description="Bill statement amount in Aug")
    BILL_AMT3: float = Field(..., description="Bill statement amount in Jul")
    BILL_AMT4: float = Field(..., description="Bill statement amount in Jun")
    BILL_AMT5: float = Field(..., description="Bill statement amount in May")
    BILL_AMT6: float = Field(..., description="Bill statement amount in Apr")
    PAY_AMT1: float = Field(..., description="Previous payment in Sep (NT dollar)")
    PAY_AMT2: float = Field(..., description="Previous payment in Aug")
    PAY_AMT3: float = Field(..., description="Previous payment in Jul")
    PAY_AMT4: float = Field(..., description="Previous payment in Jun")
    PAY_AMT5: float = Field(..., description="Previous payment in May")
    PAY_AMT6: float = Field(..., description="Previous payment in Apr")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "LIMIT_BAL": 200000.0,
                    "SEX": 2,
                    "EDUCATION": 2,
                    "MARRIAGE": 1,
                    "AGE": 35,
                    "PAY_0": 0,
                    "PAY_2": 0,
                    "PAY_3": -1,
                    "PAY_4": -1,
                    "PAY_5": -1,
                    "PAY_6": -1,
                    "BILL_AMT1": 50000.0,
                    "BILL_AMT2": 48000.0,
                    "BILL_AMT3": 45000.0,
                    "BILL_AMT4": 42000.0,
                    "BILL_AMT5": 40000.0,
                    "BILL_AMT6": 38000.0,
                    "PAY_AMT1": 5000.0,
                    "PAY_AMT2": 4800.0,
                    "PAY_AMT3": 4500.0,
                    "PAY_AMT4": 4200.0,
                    "PAY_AMT5": 4000.0,
                    "PAY_AMT6": 3800.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default")
    risk_label: str = Field(..., description="Risk category: 'Low Risk', 'Medium Risk', or 'High Risk'")
    risk_segment: Optional[str] = Field(None, description="Cluster-based risk segment")
    threshold_used: float = Field(..., description="Classification threshold used")
    model_used: str = Field(..., description="Name of the model that produced the prediction")


class ExplanationResponse(BaseModel):
    """Response schema for an explained prediction (SHAP-based)."""

    prediction: PredictionResponse
    top_risk_factors: List[Dict] = Field(
        ...,
        description="Top features driving risk up: [{feature, value, shap_value, direction}]",
    )
    top_protective_factors: List[Dict] = Field(
        ...,
        description="Top features driving risk down: [{feature, value, shap_value, direction}]",
    )
    risk_explanation: str = Field(..., description="Human-readable explanation of the prediction")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    customers: List[CustomerData] = Field(..., min_length=1, description="List of customers to predict")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    summary: Dict = Field(
        ...,
        description="Summary stats: {total, high_risk_count, high_risk_percentage}",
    )


class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint."""

    status: str = Field(..., description="Service status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_type: str = Field(..., description="Type of loaded model (e.g. 'XGBClassifier')")
    version: str = Field(..., description="API version")


class DriftReport(BaseModel):
    """Response schema for data-drift monitoring."""

    is_drifted: bool = Field(..., description="Whether significant drift was detected")
    drift_score: float = Field(..., description="Aggregate drift score (mean p-value)")
    drifted_features: List[str] = Field(..., description="Features with detected drift")
    report_timestamp: str = Field(..., description="ISO-8601 timestamp of the report")
