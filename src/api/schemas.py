"""Pydantic models that define and validate the API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """One patient record. Ranges mirror the UCI data dictionary."""
    age: float = Field(..., ge=0, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female", examples=[1])
    cp: int = Field(..., ge=0, le=4, description="chest pain type", examples=[3])
    trestbps: float = Field(..., ge=50, le=260, description="resting BP (mm Hg)", examples=[145])
    chol: float = Field(..., ge=80, le=700, description="cholesterol (mg/dl)", examples=[233])
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120", examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[0])
    thalach: float = Field(..., ge=50, le=250, description="max heart rate", examples=[150])
    exang: int = Field(..., ge=0, le=1, description="exercise-induced angina", examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=0, le=2, examples=[0])
    ca: int = Field(..., ge=0, le=4, examples=[0])
    thal: int = Field(..., ge=0, le=7, examples=[1])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = disease predicted, 0 = not")
    probability: float = Field(..., description="probability of class 1")
    label: str
    model_family: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_family: str | None = None
