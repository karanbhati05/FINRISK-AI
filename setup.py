"""FinRisk AI - Financial Risk Prediction ML Platform."""

from setuptools import setup, find_packages

setup(
    name="finrisk",
    version="0.1.0",
    description="Financial Risk Prediction ML Platform",
    author="FinRisk AI Team",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "pyyaml>=6.0",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "mlflow>=2.8",
        "pydantic>=2.5",
        "shap>=0.42",
        "imbalanced-learn>=0.11",
        "evidently>=0.4",
    ],
)
