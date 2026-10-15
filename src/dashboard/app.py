import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data.synthetic import generate_synthetic_data
from src.models.dml_estimator import DoubleMLEstimator
from src.optimization.knapsack import solve_knapsack_discount_allocation
from src.optimization.policy_allocator import compare_allocation_policies
from src.evaluation.metrics import compute_uplift_curve, compute_qini_score, summarize_causal_metrics

st.set_page_config(
    page_title="EconoCausal | Uplift & Discount Optimizer",
    page_icon="📈",
    layout="wide"
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 EconoCausal: Causal Inference & Discount Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Double Machine Learning (DML) for Heterogeneous Treatment Effects & Knapsack Budget Allocation</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Simulation & Budget Settings")
n_samples = st.sidebar.slider("Number of Customers", min_value=1000, max_value=20000, value=5000, step=1000)
budget = st.sidebar.slider("Discount Budget ($)", min_value=1000, max_value=50000, value=10000, step=1000)
discount_rate = st.sidebar.slider("Discount Rate (%)", min_value=5, max_value=30, value=15, step=1) / 100.0
nuisance_type = st.sidebar.selectbox("First-Stage Nuisance Model", ["random_forest", "linear"])

@st.cache_data(show_spinner="Generating dataset and fitting Double ML...")
def run_pipeline(n_samples: int, nuisance_type: str):
    df = generate_synthetic_data(n_samples=n_samples, random_state=42)
    feature_cols = ["Age", "PriorSpend", "VisitFrequency", "LoyaltyScore"]
    
    dml = DoubleMLEstimator(n_splits=3, nuisance_type=nuisance_type, backend="native", random_state=42)
    dml.fit(df[feature_cols], df["Discount"], df["Sales"])
    df["Estimated_ITE"] = dml.ite_estimates_
    return df, dml

data_df, dml_model = run_pipeline(n_samples, nuisance_type)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Executive Summary",
    "📊 Data & Confounding",
    "🧠 Double ML Estimates",
    "💼 Budget Optimization",
    "📈 Uplift & Qini Curves"
])

with tab1:
    st.markdown("### Problem Statement & Causal Approach")
    st.write(
        """
        Giving discounts to everyone is inefficient:
        - **Sure Things**: Loyal customers who would buy even without a discount.
        - **Lost Causes**: Customers who will never buy, even if given a discount.
        - **Persuadables**: Customers who buy **only** when given a discount.
        
        **EconoCausal** uses Double Machine Learning to isolate the causal incremental uplift (**Individual Treatment Effect, ITE**)
        free from confounding bias, and then solves a 0-1 Knapsack problem to target the optimal subset of customers under a hard budget cap.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Treatment Effect (ATE)", f"${dml_model.ate_:.2f}", help="Mean incremental sales per discount across all customers")
    col2.metric("Standard Error", f"${dml_model.standard_error_:.3f}")
    col3.metric("Total Customers", f"{len(data_df):,}")
    col4.metric("Simulated Budget", f"${budget:,.0f}")

with tab2:
    st.markdown("### Customer Features & Confounder Distribution")
    st.dataframe(data_df.head(10), use_container_width=True)
    
    fig_scatter = px.scatter(
        data_df.sample(min(1000, len(data_df))),
        x="PriorSpend",
        y="Sales",
        color="Discount",
        title="Sales vs Prior Spend (Colored by Discount)",
        color_continuous_scale="Viridis",
        opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("### Estimated Individual Treatment Effect (ITE) Distribution")
    fig_hist = px.histogram(
        data_df,
        x="Estimated_ITE",
        nbins=40,
        title="Heterogeneity in Customer Uplift (Incremental Sales per Discount)",
        color_discrete_sequence=["#2563EB"]
    )
    fig_hist.add_vline(x=dml_model.ate_, line_dash="dash", line_color="red", annotation_text=f"ATE = ${dml_model.ate_:.2f}")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.markdown("### Budget-Constrained Policy Allocation (Knapsack)")
    comparison_table = compare_allocation_policies(
        data_df,
        data_df["Estimated_ITE"].values,
        budget=float(budget),
        discount_rate=discount_rate
    )
    st.table(comparison_table)
    
    fig_bar = px.bar(
        comparison_table,
        x="Policy",
        y="Incremental_Gain",
        color="Policy",
        title=f"Total Incremental Revenue Under Budget ${budget:,.0f}",
        text_auto=".2s"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab5:
    st.markdown("### Causal Uplift & Qini Evaluation")
    curve_data = compute_uplift_curve(
        data_df["Sales"].values,
        data_df["Discount"].values,
        data_df["Estimated_ITE"].values,
        n_bins=10
    )
    qini_score = compute_qini_score(curve_data)
    st.metric("Area Under Qini Curve (AUUC)", f"{qini_score:,.1f}")
    
    fig_qini = go.Figure()
    fig_qini.add_trace(go.Scatter(
        x=curve_data["FractionTargeted"] * 100,
        y=curve_data["CumulativeQini"],
        mode="lines+markers",
        name="EconoCausal Policy",
        line=dict(color="#1D4ED8", width=3)
    ))
    fig_qini.add_trace(go.Scatter(
        x=curve_data["FractionTargeted"] * 100,
        y=curve_data["RandomBaseline"],
        mode="lines",
        name="Random Targeting Baseline",
        line=dict(color="#DC2626", dash="dash")
    ))
    fig_qini.update_layout(
        title="Qini Curve: Cumulative Incremental Sales Captured",
        xaxis_title="% of Customers Targeted",
        yaxis_title="Cumulative Incremental Sales ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_qini, use_container_width=True)
