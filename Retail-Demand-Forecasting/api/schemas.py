"""Pydantic request/response models for the Retail Demand Forecasting API."""

from __future__ import annotations

from datetime import date as date_type

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """A single-point prediction request: one (store, item, date) triple."""

    store: int = Field(..., ge=1, description="Store ID")
    item: int = Field(..., ge=1, description="Item ID")
    date: date_type = Field(..., description="Target date to forecast sales for")


class PredictResponse(BaseModel):
    """Prediction result for a single (store, item, date) request."""

    store: int
    item: int
    date: date_type
    predicted_sales: float
    model_used: str


class ForecastRequest(BaseModel):
    """A full-horizon forecast request for one (store, item) series."""

    store: int = Field(..., ge=1, description="Store ID")
    item: int = Field(..., ge=1, description="Item ID")
    horizon_days: int = Field(90, ge=1, le=90, description="Number of days to forecast (max 90)")


class ForecastPoint(BaseModel):
    """One day's forecast within a multi-day forecast response."""

    date: date_type
    predicted_sales: float


class ForecastResponse(BaseModel):
    """Full-horizon forecast result for one (store, item) series."""

    store: int
    item: int
    model_used: str
    forecast: list[ForecastPoint]


class ModelInfoResponse(BaseModel):
    """Metadata about the currently served model and its measured accuracy."""

    model_name: str
    feature_count: int
    validation_metrics: dict[str, float]
    available_models: list[str]


class HealthResponse(BaseModel):
    """Service health/readiness status."""

    status: str
    model_loaded: bool
    model_name: str | None = None
