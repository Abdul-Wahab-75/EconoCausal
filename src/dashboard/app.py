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

# Page Configuration
st.set_page_config(
    page_title="EconoCausal | Executive Promotion Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Executive Visual Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.25);
        color: #93C5FD;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
        border: 1px solid rgba(147, 197, 253, 0.3);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 12px;
        margin-top: 8px;
        font-weight: 600;
    }
    .text-green { color: #16A34A; }
    .text-blue { color: #2563EB; }
    .text-amber { color: #D97706; }
    .text-purple { color: #7C3AED; }

    /* Section Containers */
    .chart-box {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .chart-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .chart-desc {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 12px;
    }
    .insight-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Enterprise Causal AI & Budget Allocation</div>
    <h1 class="hero-title">EconoCausal: Optimal Promotion Intelligence</h1>
    <p class="hero-subtitle">Double Machine Learning (DML) identifies customers who buy <i>only</i> due to discounts; 0-1 Knapsack allocates budget for maximum ROI.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Executive Control Panel")

dataset_type = st.sidebar.selectbox(
    "Data Source",
    [
        "Real-World Rossmann Store Sales (1,115 Stores)",
        "Synthetic Benchmark (Known Ground Truth)"
    ],
    index=0
)

if dataset_type == "Synthetic Benchmark (Known Ground Truth)":
    sample_size = st.sidebar.slider("Customer Population", min_value=1000, max_value=15000, value=5000, step=1000)
    budget = st.sidebar.slider("Marketing Budget ($)", min_value=1000, max_value=50000, value=10000, step=1000)
    discount_rate = st.sidebar.slider("Discount Value (%)", min_value=5, max_value=30, value=15, step=1) / 100.0
    nuisance_choice = st.sidebar.selectbox("AI Estimator", ["random_forest", "linear"], index=0)
else:
    sample_size = st.sidebar.slider("Store-Day Records", min_value=2000, max_value=15000, value=5000, step=1000)
    budget = st.sidebar.slider("Promotion Budget ($)", min_value=5000, max_value=100000, value=25000, step=5000)
    discount_rate = st.sidebar.slider("Promo Cost Rate (%)", min_value=2, max_value=15, value=5, step=1) / 100.0
    nuisance_choice = st.sidebar.selectbox("AI Estimator", ["linear", "random_forest"], index=0)


# Cached Pipeline Engine
@st.cache_data(show_spinner="Training Double Machine Learning model & solving Knapsack...")
def execute_causal_engine(dataset_choice: str, n_records: int, model_type: str, budget_val: float, disc_rate: float):
    if dataset_choice == "Synthetic Benchmark (Known Ground Truth)":
        df = generate_synthetic_data(n_samples=n_records, random_state=42)
        feat_cols = ["Age", "PriorSpend", "VisitFrequency", "LoyaltyScore"]
        dml = DoubleMLEstimator(n_splits=3, nuisance_type=model_type, backend="native", random_state=42)
        dml.fit(df[feat_cols], df["Discount"], df["Sales"])
        df["Estimated_ITE"] = dml.ite_estimates_
        
        # Calculate naive vs causal
        naive_treated = df[df["Discount"] == 1]["Sales"].mean()
        naive_control = df[df["Discount"] == 0]["Sales"].mean()
        naive_lift = naive_treated - naive_control
        
        # Knapsack
        knapsack_res = solve_knapsack_discount_allocation(
            ite_estimates=df["Estimated_ITE"].values,
            item_prices=df["EstimatedItemPrice"].values,
            budget=budget_val,
            discount_rate=disc_rate,
            method="milp"
        )
        df["Decision"] = np.where(knapsack_res["selected_flags"] == 1, "TARGET", "SKIP")
        
        # Policy comparison
        policy_df = compare_allocation_policies(df, df["Estimated_ITE"].values, budget=budget_val, discount_rate=disc_rate)
        
        # Archetypes
        # Persuadables = High ITE; Sure Things = High Sales + Low ITE; Lost Causes = Low Sales + Low ITE
        median_sales = df["Sales"].median()
        median_ite = df["Estimated_ITE"].median()
        
        archetypes = []
        for _, row in df.iterrows():
            if row["Estimated_ITE"] > median_ite:
                archetypes.append("Persuadables (Targeted)")
            elif row["Sales"] > median_sales:
                archetypes.append("Sure Things (Skipped)")
            else:
                archetypes.append("Lost Causes (Skipped)")
        df["Archetype"] = archetypes
        
        return df, dml, naive_lift, knapsack_res, policy_df, feat_cols
    else:
        full_df = load_rossmann()
        df = full_df.sample(min(n_records, len(full_df)), random_state=42).copy()
        df = engineer_features(df)
        feat_cols = get_rossmann_feature_cols(df)
        df["EstimatedItemPrice"] = np.maximum(df["Sales"] * 0.10, 15.0)
        
        dml = DoubleMLEstimator(n_splits=3, nuisance_type=model_type, backend="native", random_state=42)
        dml.fit(df[feat_cols], df["Discount"], df["Sales"])
        df["Estimated_ITE"] = dml.ite_estimates_
        
        naive_treated = df[df["Discount"] == 1]["Sales"].mean()
        naive_control = df[df["Discount"] == 0]["Sales"].mean()
        naive_lift = naive_treated - naive_control
        
        knapsack_res = solve_knapsack_discount_allocation(
            ite_estimates=df["Estimated_ITE"].values,
            item_prices=df["EstimatedItemPrice"].values,
            budget=budget_val,
            discount_rate=disc_rate,
            method="milp"
        )
        df["Decision"] = np.where(knapsack_res["selected_flags"] == 1, "TARGET", "SKIP")
        policy_df = compare_allocation_policies(df, df["Estimated_ITE"].values, budget=budget_val, discount_rate=disc_rate)
        
        median_sales = df["Sales"].median()
        median_ite = df["Estimated_ITE"].median()
        archetypes = []
        for _, row in df.iterrows():
            if row["Estimated_ITE"] > median_ite:
                archetypes.append("High Uplift Stores (Targeted)")
            elif row["Sales"] > median_sales:
                archetypes.append("Natural Flagship Days (Skipped)")
            else:
                archetypes.append("Low Traffic Baseline (Skipped)")
        df["Archetype"] = archetypes
        
        return df, dml, naive_lift, knapsack_res, policy_df, feat_cols

# Run Engine
data_df, model_dml, naive_lift_val, knapsack_out, policy_comp_df, active_feats = execute_causal_engine(
    dataset_type, sample_size, nuisance_choice, float(budget), discount_rate
)

# -------------------------------------------------------------
# 1. EXECUTIVE KPI BANNER
# -------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

budget_spent = knapsack_out["total_cost"]
total_incremental = knapsack_out["total_incremental_value"]
roi_val = knapsack_out["estimated_roi"]
targeted_count = knapsack_out["n_targeted"]
total_count = len(data_df)
skipped_count = total_count - targeted_count
wasted_saved = skipped_count * (budget / max(targeted_count, 1)) * 0.25

with kpi_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Allocated Budget</div>
        <div class="kpi-value text-blue">${budget_spent:,.0f}</div>
        <div class="kpi-subtext text-blue">100% On-Budget (Cap: ${budget:,.0f})</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Incremental Revenue Lift</div>
        <div class="kpi-value text-green">+${total_incremental:,.0f}</div>
        <div class="kpi-subtext text-green">Pure causal profit unlocked by AI</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Campaign ROI</div>
        <div class="kpi-value text-purple">{roi_val:.1f}x</div>
        <div class="kpi-subtext text-purple">For every $1 discount spent</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Wasted Budget Avoided</div>
        <div class="kpi-value text-amber">{skipped_count:,} Skipped</div>
        <div class="kpi-subtext text-amber">Margin saved from loyal buyers</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# 2. SIDE-BY-SIDE VISUAL PROOF (THE "AHA!" & TOURNAMENT)
# -------------------------------------------------------------
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown("""
    <div class="chart-box">
        <div class="chart-title">💡 The Confounding Trap: Correlation vs. True Causation</div>
        <div class="chart-desc">Traditional dashboards credit natural foot traffic to promos. DML isolates true lift.</div>
    """, unsafe_allow_html=True)
    
    bias_val = max(naive_lift_val - model_dml.ate_, 0.0)
    fig_trap = go.Figure()
    
    fig_trap.add_trace(go.Bar(
        name="True Causal Lift (Our AI)",
        x=["Promotion Effect"],
        y=[model_dml.ate_],
        marker_color="#2563EB",
        text=[f"+${model_dml.ate_:.0f}"],
        textposition="auto"
    ))
    fig_trap.add_trace(go.Bar(
        name="Confounding (Foot Traffic / Calendar)",
        x=["Promotion Effect"],
        y=[bias_val],
        marker_color="#F59E0B",
        text=[f"Bias: -${bias_val:.0f}"],
        textposition="auto"
    ))
    fig_trap.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        yaxis_title="Sales Lift per Day ($)"
    )
    st.plotly_chart(fig_trap, use_container_width=True)
    
    pct_bias = (bias_val / max(naive_lift_val, 1e-6)) * 100
    st.markdown(f"""
    <div class="insight-pill">
        ⚡ <b>Key Finding:</b> {pct_bias:.0f}% of raw promotional sales was organic demand that would have happened anyway!
    </div>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown(f"""
    <div class="chart-box">
        <div class="chart-title">🏆 Head-to-Head Strategy Tournament</div>
        <div class="chart-desc">Comparing incremental revenue under identical ${budget:,.0f} budget.</div>
    """, unsafe_allow_html=True)
    
    colors = ["#16A34A", "#F59E0B", "#94A3B8"]
    fig_tourney = px.bar(
        policy_comp_df,
        x="Policy",
        y="Incremental_Gain",
        color="Policy",
        color_discrete_sequence=colors,
        text_auto=".2s"
    )
    fig_tourney.update_layout(
        height=320,
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=20),
        template="plotly_white",
        yaxis_title="Incremental Revenue ($)"
    )
    st.plotly_chart(fig_tourney, use_container_width=True)
    
    causal_gain = policy_comp_df.iloc[0]["Incremental_Gain"]
    heuristic_gain = max(policy_comp_df.iloc[1]["Incremental_Gain"], 1.0)
    multiple = causal_gain / heuristic_gain
    st.markdown(f"""
    <div class="insight-pill">
        🚀 <b>Outperformance:</b> EconoCausal generates <b>{multiple:.1f}x more revenue</b> than traditional high-spender targeting!
    </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. CUSTOMER ARCHETYPES & QINI EVALUATION CURVE
# -------------------------------------------------------------
row3_col1, row3_col2 = st.columns([1, 1])

with row3_col1:
    st.markdown("""
    <div class="chart-box">
        <div class="chart-title">👥 Population Segmentation: Who Gets Discounted?</div>
        <div class="chart-desc">Smart targeting filters out Sure Things to focus strictly on Persuadables.</div>
    """, unsafe_allow_html=True)
    
    arch_counts = data_df["Archetype"].value_counts().reset_index()
    arch_counts.columns = ["Segment", "Count"]
    
    fig_donut = px.pie(
        arch_counts,
        names="Segment",
        values="Count",
        hole=0.55,
        color_discrete_sequence=["#2563EB", "#F59E0B", "#94A3B8"]
    )
    fig_donut.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        template="plotly_white"
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row3_col2:
    st.markdown("""
    <div class="chart-box">
        <div class="chart-title">📈 Qini Uplift Curve (Cumulative Gain)</div>
        <div class="chart-desc">Steeper slope means faster revenue capture with fewer promotional dollars.</div>
    """, unsafe_allow_html=True)
    
    curve_data = compute_uplift_curve(
        data_df["Sales"].values,
        data_df["Discount"].values,
        data_df["Estimated_ITE"].values,
        n_bins=10
    )
    qini_score = compute_qini_score(curve_data)
    
    fig_qini = go.Figure()
    fig_qini.add_trace(go.Scatter(
        x=curve_data["FractionTargeted"] * 100,
        y=curve_data["CumulativeQini"],
        mode="lines+markers",
        name="EconoCausal AI",
        line=dict(color="#16A34A", width=3)
    ))
    fig_qini.add_trace(go.Scatter(
        x=curve_data["FractionTargeted"] * 100,
        y=curve_data["RandomBaseline"],
        mode="lines",
        name="Random Guessing",
        line=dict(color="#EF4444", dash="dash")
    ))
    fig_qini.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="% of Candidates Targeted",
        yaxis_title="Cumulative Uplift ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_qini, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. ACTIONABLE OPERATIONAL LIST & 1-CLICK CSV EXPORT
# -------------------------------------------------------------
st.markdown("""
<div class="chart-box">
    <div class="chart-title">📋 Operational Action List: Who to Contact</div>
    <div class="chart-desc">Downloadable decision table generated by the Knapsack Optimizer. Ready for marketing execution.</div>
""", unsafe_allow_html=True)

action_cols = ["CustomerID", "Sales", "Estimated_ITE", "Archetype", "Decision"]
display_df = data_df[action_cols].copy()
display_df = display_df.sort_values(by="Estimated_ITE", ascending=False).reset_index(drop=True)

# Format for clean display
display_df.rename(columns={
    "CustomerID": "Candidate ID",
    "Sales": "Historical Sales ($)",
    "Estimated_ITE": "Predicted Uplift ($)",
    "Decision": "Knapsack Allocation"
}, inplace=True)

col_tbl, col_dl = st.columns([4, 1])
with col_tbl:
    st.dataframe(
        display_df.head(15).style.map(
            lambda v: "color: #16A34A; font-weight: bold;" if v == "TARGET" else "color: #94A3B8;",
            subset=["Knapsack Allocation"]
        ),
        use_container_width=True,
        height=260
    )

with col_dl:
    st.write("")
    st.write("")
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Action CSV",
        data=csv_bytes,
        file_name="EconoCausal_Targeting_Action_List.csv",
        mime="text/csv",
        help="Export full target list ready for your email, SMS, or retail marketing platform.",
        use_container_width=True
    )
    st.success(f"✅ **{targeted_count:,}** selected\n\n❌ **{skipped_count:,}** skipped")

st.markdown("</div>", unsafe_allow_html=True)
