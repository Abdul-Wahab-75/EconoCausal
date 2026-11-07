import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_data
from src.models.dowhy_analysis import (
    DEFAULT_CONFOUNDERS,
    DEFAULT_REFUTERS,
    build_causal_graph,
    run_dowhy_analysis,
)


def test_build_causal_graph_contains_required_relationships():
    graph = build_causal_graph(
        confounders=DEFAULT_CONFOUNDERS,
        treatment="Discount",
        outcome="Sales",
    )

    for confounder in DEFAULT_CONFOUNDERS:
        assert f"{confounder} -> Discount;" in graph
        assert f"{confounder} -> Sales;" in graph

    assert "Discount -> Sales;" in graph


def test_dowhy_analysis_estimates_effect_and_runs_refutations():
    df = generate_synthetic_data(
        n_samples=500,
        random_state=42,
    )

    result = run_dowhy_analysis(df)

    assert np.isfinite(result["ate"])
    assert result["treatment"] == "Discount"
    assert result["outcome"] == "Sales"
    assert result["confounders"] == DEFAULT_CONFOUNDERS

    assert set(result["refutations"]) == set(DEFAULT_REFUTERS)

    for refutation in result["refutations"].values():
        assert refutation is not None


def test_dowhy_analysis_rejects_missing_columns():
    df = pd.DataFrame({
        "Discount": [0, 1, 0, 1],
        "Sales": [100, 120, 110, 130],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        run_dowhy_analysis(df)


def test_dowhy_analysis_rejects_missing_values():
    df = generate_synthetic_data(
        n_samples=100,
        random_state=42,
    )
    df.loc[0, "Age"] = np.nan

    with pytest.raises(ValueError, match="must not contain missing values"):
        run_dowhy_analysis(df)
