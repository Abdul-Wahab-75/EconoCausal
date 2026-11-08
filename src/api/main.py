from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from src.evaluation.metrics import compute_uplift_curve, compute_qini_score
from src.optimization.knapsack import solve_knapsack_discount_allocation

from src.models.dml_estimator import DoubleMLEstimator
from src.evaluation.drift import detect_data_drift
from src.models.propensity_matching import propensity_score_matching
from src.models.dowhy_analysis import run_dowhy_analysis
from src.optimization.policy_allocator import compare_allocation_policies
from src.dashboard.strategy_summary import create_strategy_summary


app = FastAPI(
    title="EconoCausal REST API",
    description="REST API for the Double Machine Learning causal inference engine.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model state for the running API process.
estimator: Optional[DoubleMLEstimator] = None
trained_feature_names: List[str] = []
training_observations: int = 0


class FitRequest(BaseModel):
    feature_names: List[str] = Field(min_length=1)
    features: List[List[float]] = Field(min_length=2)
    treatment: List[float] = Field(min_length=2)
    outcome: List[float] = Field(min_length=2)
    n_splits: int = Field(default=3, ge=2)
    nuisance_type: str = Field(default="random_forest")
    random_state: int = 42


class PredictRequest(BaseModel):
    feature_names: List[str] = Field(min_length=1)
    features: List[List[float]] = Field(min_length=1)

class UpliftRequest(BaseModel):
    treatment: List[float] = Field(min_length=10)
    outcome: List[float] = Field(min_length=10)
    ite_predictions: List[float] = Field(min_length=10)

class OptimizeRequest(BaseModel):
    ite_estimates: List[float] = Field(min_length=1)
    item_prices: List[float] = Field(min_length=1)
    budget: float = Field(gt=0)
    discount_rate: float = Field(default=0.15, gt=0, le=1) 
class StrategySummaryRequest(BaseModel):
    ite_estimates: List[float] = Field(min_length=1)
    item_prices: List[float] = Field(min_length=1)
    budget: float = Field(gt=0)
    discount_rate: float = Field(default=0.15, gt=0, le=1)
    dataset_name: str = "Rossmann"
    prior_spend: Optional[List[float]] = None

class PSMRequest(BaseModel):
    feature_names: List[str] = Field(min_length=1)
    features: List[List[float]] = Field(min_length=2)
    treatment: List[int] = Field(min_length=2)
    caliper: float = Field(default=0.2, gt=0)
    random_state: int = 42


class DoWhyRequest(BaseModel):
    treatment: str
    outcome: str
    confounders: List[str] = Field(min_length=1)
    data: List[Dict[str, float]] = Field(min_length=2)
    refutation_methods: List[str] = Field(
        default_factory=lambda: [
            "random_common_cause",
            "placebo_treatment_refuter",
            "data_subset_refuter",
        ]
    )    


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_fitted": estimator is not None,
    }


@app.post("/fit")
def fit_model(request: FitRequest):
    global estimator, trained_feature_names, training_observations

    if len(set(request.feature_names)) != len(request.feature_names):
        raise HTTPException(status_code=400, detail="Feature names must be unique.")

    if request.nuisance_type not in {
        "random_forest",
        "gradient_boosting",
        "linear",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported nuisance_type. Use random_forest, gradient_boosting, or linear.",
        )

    if len(request.features) != len(request.treatment) or len(request.features) != len(request.outcome):
        raise HTTPException(
            status_code=400,
            detail="Features, treatment, and outcome must contain the same number of rows.",
        )

    if len(request.features) < request.n_splits:
        raise HTTPException(
            status_code=400,
            detail="Number of observations must be at least equal to n_splits.",
        )

    if any(len(row) != len(request.feature_names) for row in request.features):
        raise HTTPException(
            status_code=400,
            detail="Every feature row must match the number of feature_names.",
        )

    if not all(value in (0, 1) for value in request.treatment):
        raise HTTPException(
            status_code=400,
            detail="Treatment values must be binary (0 or 1).",
        )

    if len(set(request.treatment)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Treatment must contain both 0 and 1 observations.",
        )

    try:
        X = pd.DataFrame(request.features, columns=request.feature_names)
        D = pd.Series(request.treatment, name="Treatment")
        Y = pd.Series(request.outcome, name="Outcome")

        new_estimator = DoubleMLEstimator(
            n_splits=request.n_splits,
            nuisance_type=request.nuisance_type,
            backend="econml",
            random_state=request.random_state,
        )
        new_estimator.fit(X, D, Y)

        estimator = new_estimator
        trained_feature_names = request.feature_names
        training_observations = len(X)

        return {
            "status": "success",
            "message": "Causal inference model fitted successfully.",
            "observations": training_observations,
            "feature_names": trained_feature_names,
            "ate": float(estimator.ate_),
            "standard_error": float(estimator.standard_error_),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Model fitting failed: {str(exc)}",
        ) from exc


@app.post("/predict")
def predict_ite(request: PredictRequest):
    if estimator is None:
        raise HTTPException(
            status_code=409,
            detail="Model is not fitted. Call POST /fit first.",
        )

    if request.feature_names != trained_feature_names:
        raise HTTPException(
            status_code=400,
            detail="Feature names and order must match the training data.",
        )

    if any(len(row) != len(trained_feature_names) for row in request.features):
        raise HTTPException(
            status_code=400,
            detail="Every feature row must match the trained feature count.",
        )

    try:
        X = pd.DataFrame(request.features, columns=request.feature_names)
        predictions = estimator.predict_ite(X)

        return {
            "status": "success",
            "predictions": [float(value) for value in predictions],
            "count": len(predictions),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"ITE prediction failed: {str(exc)}",
        ) from exc


@app.get("/model-info")
def model_info():
    if estimator is None:
        return {
            "fitted": False,
            "message": "No trained model is available. Call POST /fit first.",
        }

    return {
        "fitted": True,
        "backend": (
    "econml"
    if estimator._econml_model is not None
    else "native"
),
        "observations": training_observations,
        "feature_names": trained_feature_names,
        "ate": float(estimator.ate_),
        "standard_error": float(estimator.standard_error_),
    }


class DriftRequest(BaseModel):
    reference: List[Dict[str, float]] = Field(min_length=2)
    current: List[Dict[str, float]] = Field(min_length=2)
    psi_threshold: float = Field(default=0.20, ge=0)
    significance_level: float = Field(default=0.05, gt=0, lt=1)


@app.post("/drift")
def check_data_drift(request: DriftRequest):
    try:
        reference_df = pd.DataFrame(request.reference)
        current_df = pd.DataFrame(request.current)

        result = detect_data_drift(
            reference=reference_df,
            current=current_df,
            psi_threshold=request.psi_threshold,
            significance_level=request.significance_level,
        )

        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Data drift detection failed: {str(exc)}",
        ) from exc
@app.post("/uplift")
def evaluate_uplift(request: UpliftRequest):
    if not (
        len(request.treatment)
        == len(request.outcome)
        == len(request.ite_predictions)
    ):
        raise HTTPException(
            status_code=400,
            detail="Treatment, outcome and ITE arrays must have equal lengths.",
        )

    if set(request.treatment) != {0, 1}:
        raise HTTPException(
            status_code=400,
            detail="Treatment must contain both 0 and 1.",
        )

    curve = compute_uplift_curve(
        y_true=request.outcome,
        treatment=request.treatment,
        ite_preds=request.ite_predictions,
    )

    qini_score = compute_qini_score(curve)

    records = [
        {key: float(value) for key, value in row.items()}
        for row in curve.to_dict(orient="records")
    ]

    return {
        "status": "success",
        "qini_score": float(qini_score),
        "uplift_curve": records,
    }

@app.post("/optimize")
def optimize_budget(request: OptimizeRequest):
    if len(request.ite_estimates) != len(request.item_prices):
        raise HTTPException(
            status_code=400,
            detail="ITE estimates and item prices must have equal lengths.",
        )

    if any(price <= 0 for price in request.item_prices):
        raise HTTPException(
            status_code=400,
            detail="All item prices must be greater than zero.",
        )

    try:
        result = solve_knapsack_discount_allocation(
            ite_estimates=request.ite_estimates,
            item_prices=request.item_prices,
            budget=request.budget,
            discount_rate=request.discount_rate,
            method="milp",
            ite_is_revenue=False,
        )

        return {
            "status": "success",
            "selected_flags": [
                bool(value) for value in result["selected_flags"]
            ],
            "targeted_customers": int(result["n_targeted"]),
            "total_cost": float(result["total_cost"]),
            "budget_limit": float(result["budget_limit"]),
            "total_incremental_value": float(
                result["total_incremental_value"]
            ),
            "estimated_roi": float(result["estimated_roi"]),
            "customer_values": [
                float(value) for value in result["customer_values"]
            ],
            "customer_costs": [
                float(value) for value in result["customer_costs"]
            ],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Budget optimization failed: {str(error)}",
        ) from error


@app.post("/psm")
def run_psm(request: PSMRequest):
    if len(request.features) != len(request.treatment):
        raise HTTPException(
            status_code=400,
            detail="Features and treatment must contain the same number of rows.",
        )

    if any(len(row) != len(request.feature_names) for row in request.features):
        raise HTTPException(
            status_code=400,
            detail="Each feature row must match the feature_names length.",
        )

    if len(set(request.feature_names)) != len(request.feature_names):
        raise HTTPException(
            status_code=400,
            detail="Feature names must be unique.",
        )

    try:
        X = pd.DataFrame(
            request.features,
            columns=request.feature_names,
        )
        treatment = pd.Series(request.treatment)

        result = propensity_score_matching(
            X=X,
            treatment=treatment,
            caliper=request.caliper,
            random_state=request.random_state,
        )

        return {
            "status": "success",
            "propensity_scores": result["propensity_scores"].tolist(),
            "matched_pairs": result["matched_pairs"].to_dict(orient="records"),
            "balance_before": result["balance_before"].to_dict(orient="records"),
            "balance_after": result["balance_after"].to_dict(orient="records"),
            "treated_before": result["n_treated_before"],
            "control_before": result["n_control_before"],
            "treated_matched": result["n_treated_matched"],
            "control_matched": result["n_control_matched"],
            "caliper_width": result["caliper_width"],
            "common_support": list(result["common_support"]),
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PSM analysis failed: {str(exc)}",
        ) from exc
@app.post("/dowhy")
def run_dowhy(request: DoWhyRequest):
    try:
        data = pd.DataFrame(request.data)

        result = run_dowhy_analysis(
            data=data,
            treatment=request.treatment,
            outcome=request.outcome,
            confounders=request.confounders,
            refutation_methods=request.refutation_methods,
        )

        return {
            "status": "success",
            "graph": result["graph"],
            "treatment": result["treatment"],
            "outcome": result["outcome"],
            "confounders": result["confounders"],
            "ate": result["ate"],
            "estimand": str(result["estimand"]),
            "estimate": str(result["estimate"]),
            "refutations": {
                name: str(value)
                for name, value in result["refutations"].items()
            },
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"DoWhy analysis failed: {str(exc)}",
        ) from exc   

    
@app.post("/strategy-summary")
def strategy_summary(request: StrategySummaryRequest):
    if len(request.ite_estimates) != len(request.item_prices):
        raise HTTPException(
            status_code=400,
            detail="ITE estimates and item prices must have equal lengths.",
        )

    if any(price <= 0 for price in request.item_prices):
        raise HTTPException(
            status_code=400,
            detail="All item prices must be greater than zero.",
        )

    if request.prior_spend is not None:
        if len(request.prior_spend) != len(request.ite_estimates):
            raise HTTPException(
                status_code=400,
                detail="Prior spend and ITE estimates must have equal lengths.",
            )

    try:
        data = {
            "EstimatedItemPrice": request.item_prices,
        }

        if request.prior_spend is not None:
            data["PriorSpend"] = request.prior_spend

        df = pd.DataFrame(data)

        knapsack_result = solve_knapsack_discount_allocation(
            ite_estimates=request.ite_estimates,
            item_prices=request.item_prices,
            budget=request.budget,
            discount_rate=request.discount_rate,
            method="milp",
            ite_is_revenue=False,
        )

        policy_comparison = compare_allocation_policies(
            df=df,
            ite_estimates=request.ite_estimates,
            budget=request.budget,
            discount_rate=request.discount_rate,
        )

        summary = create_strategy_summary(
            knapsack_result=knapsack_result,
            policy_comparison=policy_comparison,
            dataset_name=request.dataset_name,
            currency_symbol="₹",
        )

        return {
            "status": "success",
            "summary": summary,
            "policy_comparison": policy_comparison.to_dict(
                orient="records"
            ),
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy summary failed: {str(exc)}",
        ) from exc
