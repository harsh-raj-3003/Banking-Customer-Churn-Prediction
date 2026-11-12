"""Request and response bodies for the churn API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Geography(str, Enum):
    """Customer country, the three levels seen in training."""

    FRANCE = "France"
    GERMANY = "Germany"
    SPAIN = "Spain"


class Gender(str, Enum):
    """Customer gender."""

    FEMALE = "Female"
    MALE = "Male"


class CustomerFeatures(BaseModel):
    """One customer, with the ranges observed in the training data."""

    credit_score: int = Field(ge=300, le=850, description="Credit score (300-850)")
    geography: Geography = Field(description="Country of the customer")
    gender: Gender = Field(description="Gender of the customer")
    age: int = Field(ge=18, le=100, description="Age in years")
    tenure: int = Field(ge=0, le=10, description="Years with the bank")
    balance: float = Field(ge=0, description="Account balance")
    num_of_products: int = Field(ge=1, le=4, description="Number of bank products")
    has_cr_card: bool = Field(description="Has a credit card")
    is_active_member: bool = Field(description="Is an active member")
    estimated_salary: float = Field(ge=0, description="Estimated salary")

    def to_record(self) -> dict:
        """Convert to the raw dataset column names used by feature engineering."""
        return {
            "CreditScore": self.credit_score,
            "Geography": self.geography.value,
            "Gender": self.gender.value,
            "Age": self.age,
            "Tenure": self.tenure,
            "Balance": self.balance,
            "NumOfProducts": self.num_of_products,
            "HasCrCard": int(self.has_cr_card),
            "IsActiveMember": int(self.is_active_member),
            "EstimatedSalary": self.estimated_salary,
        }


class PredictionResponse(BaseModel):
    """Churn score for one customer."""

    churn_probability: float = Field(ge=0, le=1, description="Probability the customer leaves")
    churn_flag: bool = Field(description="1 when the probability is at or above the cut-off")
    threshold: float = Field(description="Operating cut-off used for the flag")
    model_version: str = Field(description="Identifier of the fitted model")


class HealthResponse(BaseModel):
    """Service health and model availability."""

    status: str
    model_ready: bool
    model_version: str
    threshold: float


class FeatureSpecResponse(BaseModel):
    """What the caller must send, so a UI can build the form."""

    features: list[str]
    numeric: list[str]
    binary: list[str]
    categorical: dict[str, list[str]]
    threshold: float
    model_ready: bool
