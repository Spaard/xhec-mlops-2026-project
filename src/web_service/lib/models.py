"""Pydantic models of the prediction API: request and response bodies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modelling.preprocessing import MAX_HEIGHT_RATIO, MIN_HEIGHT_RATIO

MAX_ABALONES_PER_REQUEST = 1000


class Abalone(BaseModel):
    """Physical measurements of one abalone, in the units of the Kaggle dataset.

    Besides the types, the same physical consistency rules as the cleaning of the
    training data are enforced, so the model never predicts on impossible inputs.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            }
        },
    )

    sex: Literal["M", "F", "I"] = Field(description="M (male), F (female), I (infant)")
    length: float = Field(gt=0, description="Longest shell measurement")
    diameter: float = Field(gt=0, description="Perpendicular to the length")
    height: float = Field(gt=0, description="With meat in shell")
    whole_weight: float = Field(gt=0, description="Whole abalone")
    shucked_weight: float = Field(gt=0, description="Weight of the meat")
    viscera_weight: float = Field(gt=0, description="Gut weight (after bleeding)")
    shell_weight: float = Field(gt=0, description="Shell weight, after being dried")

    @model_validator(mode="after")
    def check_physical_consistency(self) -> "Abalone":
        """Reject the measurements that no real abalone can have."""
        if self.diameter > self.length:
            raise ValueError("diameter cannot exceed length (the longest measurement)")
        for part in ("shucked_weight", "viscera_weight", "shell_weight"):
            if getattr(self, part) > self.whole_weight:
                raise ValueError(f"{part} cannot exceed whole_weight")
        height_ratio = self.height / self.length
        if not MIN_HEIGHT_RATIO <= height_ratio <= MAX_HEIGHT_RATIO:
            raise ValueError(
                f"height / length must be between {MIN_HEIGHT_RATIO} and "
                f"{MAX_HEIGHT_RATIO}, got {height_ratio:.3f}"
            )
        return self


class PredictionRequest(BaseModel):
    """Body of a prediction request: one or several abalones."""

    abalones: list[Abalone] = Field(min_length=1, max_length=MAX_ABALONES_PER_REQUEST)


class Prediction(BaseModel):
    """Predicted number of rings and age of one abalone."""

    rings: float = Field(description="Predicted number of rings")
    age: float = Field(description="Predicted age in years (rings + 1.5)")


class PredictionResponse(BaseModel):
    """Body of a prediction response, in the order of the request's abalones."""

    predictions: list[Prediction]
