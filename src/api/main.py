"""
FinRisk AI - FastAPI application entry-point.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    _model_cache,
    get_feature_engineer,
    get_model,
    get_preprocessor,
    router,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("finrisk.api")

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model artifacts on startup."""
    logger.info("Starting FinRisk AI API …")
    try:
        get_model()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning("Could not pre-load model: %s", exc)

    try:
        get_preprocessor()
        logger.info("Preprocessor loaded successfully.")
    except Exception as exc:
        logger.warning("Could not pre-load preprocessor: %s", exc)

    try:
        get_feature_engineer()
        logger.info("Feature engineer loaded successfully.")
    except Exception as exc:
        logger.warning("Could not pre-load feature engineer: %s", exc)

    yield  # app runs here

    _model_cache.clear()
    logger.info("FinRisk AI API shut down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FinRisk AI",
    description=(
        "Machine-learning API for credit-default risk prediction, "
        "SHAP-based explanations, risk segmentation, and data-drift monitoring."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request-logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Include routes
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to FinRisk AI. Visit /docs for API documentation."}
