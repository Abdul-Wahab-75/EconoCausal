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
from src.data.load_data import load_rossmann
from src.data.features import engineer_features, get_rossmann_feature_cols
from src.models.dml_estimator import DoubleMLEstimator
from src.optimization.knapsack import solve_knapsack_discount_allocation
from src.optimization.policy_allocator import compare_allocation_policies
from src.evaluation.metrics import compute_uplift_curve, compute_qini_score

st.set_page_config(
    page_title="EconoCausal | Uplift & Discount Optimizer",
    page_icon="📈",
    layout="wide"
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 15px;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 EconoCausal: Causal Inference & Discount Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Double Machine Learning (DML) for Heterogeneous Treatment Effects & 0-1 Knapsack Budget Allocation</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Simulation & Budget Settings")

dataset_type = st.sidebar.selectbox(
    "Select Dataset",
    [
        "Real-World Rossmann Store Sales",
        "Synthetic Benchmark (Known Ground Truth)"
    ],
    index=0,
    help="Switch between real retail store data (Rossmann) and simulated data with known ground-truth answers."
)

if dataset_type == "Synthetic Benchmark (Known Ground Truth)":
    n_samples = st.sidebar.slider("Number of Customers", min_value=1000, max_value=15000, value=5000, step=1000)
    budget = st.sidebar.slider("Discount Budget ($)", min_value=1000, max_value=50000, value=10000, step=1000)
    discount_rate = st.sidebar.slider("Discount Rate (%)", min_value=5, max_value=30, value=15, step=1) / 100.0
    nuisance_type = st.sidebar.selectbox("Nuisance Model", ["random_forest", "linear"], index=0)
else:
    n_samples = st.sidebar.slider("Store Days Sample", min_value=2000, max_value=20000, value=5000, step=1000)
    budget = st.sidebar.slider("Promotion Budget ($)", min_value=5000, max_value=100000, value=25000, step=5000)
    discount_rate = st.sidebar.slider("Promo Cost Rate (%)", min_value=2, max_value=15, value=5, step=1) / 100.0
    nuisance_type = st.sidebar.selectbox("Nuisance Model", ["linear", "random_forest"], index=0)


@st.cache_data(show_spinner="Loading data and training Double Machine Learning model...")
def run_pipeline(dataset_choice: str, sample_size: int, model_type: str):
    if dataset_choice == "Synthetic Benchmark (Known Ground Truth)":
        df = generate_synthetic_data(n_samples=sample_size, random_state=42)
        feature_cols = ["Age", "PriorSpend", "VisitFrequency", "LoyaltyScore"]
        dml = DoubleMLEstimator(n_splits=3, nuisance_type=model_type, backend="native", random_state=42)
        dml.fit(df[feature_cols], df["Discount"], df["Sales"])
        df["Estimated_ITE"] = dml.ite_estimates_
        naive_lift = df[df["Discount"] == 1]["Sales"].mean() - df[df["Discount"] == 0]["Sales"].mean()
        return df, dml, feature_cols, naive_lift
    else:
        full_df = load_rossmann()
        df = full_df.sample(min(sample_size, len(full_df)), random_state=42).copy()
        df = engineer_features(df)
        feature_cols = get_rossmann_feature_cols(df)
        
        # Add proxy item price for knapsack if not present
        df["EstimatedItemPrice"] = np.maximum(df["Sales"] * 0.10, 15.0)
        
        dml = DoubleMLEstimator(n_splits=3, nuisance_type=model_type, backend="native", random_state=42)
        dml.fit(df[feature_cols], df["Discount"], df["Sales"])
        df["Estimated_ITE"] = dml.ite_estimates_
        
        naive_lift = df[df["Discount"] == 1]["Sales"].mean() - df[df["Discount"] == 0]["Sales"].mean()
        return df, dml, feature_cols, naive_lift

data_df, dml_model, active_features, naive_sales_lift = run_pipeline(dataset_type, n_samples, nuisance_type)

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
        - **Sure Things**: Loyal customers who buy regardless of discounts.
        - **Lost Causes**: Customers who will not buy even with discounts.
        - **Persuadables**: Customers who buy **strictly because of** the promotion.
        
        **EconoCausal** uses Double Machine Learning (DML) to isolate the true causal **Individual Treatment Effect (ITE)**,
        free from confounding bias, and then solves a **0-1 Knapsack Optimization** to target the optimal candidates under a strict budget cap.
        """
    )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Causal Lift (ATE)", f"${dml_model.ate_:.2f}", help="True causal incremental sales per discount")
    col2.metric("Naive Observed Lift", f"${naive_sales_lift:.2f}", help="Raw average difference before causal adjustment")
    bias_removed = naive_sales_lift - dml_model.ate_
    col3.metric("Confounding Bias Removed", f"${bias_removed:.2f}", delta=f"-${abs(bias_removed):.2f}" if bias_removed > 0 else f"${abs(bias_removed):.2f}")
    col4.metric("Allocated Budget", f"${budget:,.0f}")
    
    if "TrueITE" in data_df.columns:
        true_ate = float(data_df["TrueITE"].mean())
        err = abs(true_ate - dml_model.ate_)
        st.success(f" Ground Truth Validation: True ATE = **${true_ate:.2f}** | DML Estimate = **${dml_model.ate_:.2f}** | Error = **${err:.2f}** ({(err/true_ate)*100:.1f}%)")
    else:
        st.info(f" Real Retail Data Insight: Raw sales show a **+${naive_sales_lift:.2f}** difference on promo days, but our DML engine identified that **${bias_removed:.2f}** was confounding (calendar/store effects). The true incremental lift is **+${dml_model.ate_:.2f}**.")

with tab2:
    st.markdown("### Confounder Distribution & Data Inspection")
    st.write(f"Displaying sample records with **{len(active_features)}** active confounder features:")
    preview_cols = ["CustomerID", "Discount", "Sales"] + [c for c in active_features[:5] if c in data_df.columns]
    st.dataframe(data_df[preview_cols].head(10), use_container_width=True)
    
    scatter_x = "PriorSpend" if "PriorSpend" in data_df.columns else "DayOfWeek"
    fig_scatter = px.scatter(
        data_df.sample(min(1000, len(data_df))),
        x=scatter_x,
        y="Sales",
        color="Discount",
        title=f"Sales vs {scatter_x} (Colored by Discount / Promo)",
        color_continuous_scale="Viridis",
        opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("### Heterogeneous Treatment Effect (ITE) Distribution")
    fig_hist = px.histogram(
        data_df,
        x="Estimated_ITE",
        nbins=40,
        title="Distribution of Estimated Individual Treatment Effects (ITE)",
        color_discrete_sequence=["#2563EB"]
    )
    fig_hist.add_vline(x=dml_model.ate_, line_dash="dash", line_color="red", annotation_text=f"ATE = ${dml_model.ate_:.2f}")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.markdown("### Budget-Constrained Optimization (0-1 Knapsack)")
    comparison_table = compare_allocation_policies(
        data_df,
        data_df["Estimated_ITE"].values,
        budget=float(budget),
        discount_rate=discount_rate
    )
    st.dataframe(comparison_table, use_container_width=True)
    
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
    st.markdown("### Uplift Modeling & Qini Evaluation")
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
        xaxis_title="% of Population Targeted",
        yaxis_title="Cumulative Incremental Sales ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_qini, use_container_width=True)
