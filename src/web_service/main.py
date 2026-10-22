"""Prediction API: predicts the age of abalones from their physical measurements."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from sklearn.base import BaseEstimator

from modelling.predicting import MODEL_PATH
from web_service.lib.inference import run_inference
from web_service.lib.models import PredictionRequest, PredictionResponse
from web_service.utils import load_object


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the trained model once, when the API starts."""
    app.state.model = load_object(MODEL_PATH) if MODEL_PATH.exists() else None
    yield


app = FastAPI(
    title="Abalone age prediction",
    description=(
        "Predicts the number of rings (and the age) of abalones from physical "
        "measurements that are easy to obtain."
    ),
    lifespan=lifespan,
)


def get_model(request: Request) -> BaseEstimator:
    """Return the loaded model, or fail if it has not been trained yet."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model found at {MODEL_PATH}: train it first.",
        )
    return model


@app.get("/")
def home() -> dict:
    """Health check."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionResponse, status_code=201)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict the number of rings and the age of each abalone of the request."""
    predictions = run_inference(get_model(request), payload.abalones)
    return PredictionResponse(predictions=predictions)
