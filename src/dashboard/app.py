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
from src.utils.currency import convert_currency, SUPPORTED_CURRENCIES, get_currency_meta

# Page Configuration - Clean corporate title, no emoji
st.set_page_config(
    page_title="EconoCausal | Enterprise Promotion Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS & Professional Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background override */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
    }

    /* Executive Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #111827 0%, #1E293B 60%, #0F172A 100%);
        border: 1px solid #1E293B;
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
        border: 1px solid rgba(96, 165, 250, 0.25);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 13.5px;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Executive KPI Cards */
    .kpi-card {
        background: #111827;
        border-radius: 10px;
        padding: 20px 22px;
        border: 1px solid #1F2937;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #374151;
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 12px;
        margin-top: 8px;
        font-weight: 500;
    }
    .val-blue { color: #60A5FA; }
    .val-green { color: #34D399; }
    .val-purple { color: #A78BFA; }
    .val-amber { color: #FBBF24; }
    .sub-muted { color: #64748B; }

    /* Chart Containers */
    .chart-box {
        background: #111827;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #1F2937;
        margin-bottom: 20px;
    }
    .chart-title {
        font-size: 15px;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .chart-desc {
        font-size: 12.5px;
        color: #64748B;
        margin-bottom: 14px;
    }
    .insight-pill {
        display: block;
        background: #0F172A;
        color: #93C5FD;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12px;
        font-weight: 500;
        margin-top: 12px;
        line-height: 1.4;
    }

    /* Streamlit widget tweaks */
    div[data-testid="stSidebar"] {
        background-color: #0A0E17;
        border-right: 1px solid #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# Executive Header
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Enterprise Promotion Intelligence</div>
    <h1 class="hero-title">EconoCausal: Causal Inference & Budget Allocation</h1>
    <p class="hero-subtitle">Double Machine Learning (DML) identifies customers who purchase strictly due to promotional incentives; 0-1 Knapsack Optimization allocates marketing capital to maximize incremental net return.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### Control Parameters")

# Currency Settings & Converter
st.sidebar.markdown("#### Currency Settings")
curr_list = list(SUPPORTED_CURRENCIES.keys())

data_currency = st.sidebar.selectbox(
    "Data Input Currency",
    curr_list,
    index=curr_list.index("INR"),
    format_func=lambda code: f"{code} ({SUPPORTED_CURRENCIES[code]['symbol']} - {SUPPORTED_CURRENCIES[code]['name']})",
    help="Select the native currency that your source data / transactions are recorded in."
)

target_currency = st.sidebar.selectbox(
    "Display & Output Currency",
    curr_list,
    index=curr_list.index("INR"),
    format_func=lambda code: f"{code} ({SUPPORTED_CURRENCIES[code]['symbol']} - {SUPPORTED_CURRENCIES[code]['name']})",
    help="Select which currency you want all figures, KPIs, optimization budget, and CSV exports converted to."
)

# Compute conversion multiplier: 1 unit of data_currency -> conv_rate units of target_currency
conv_rate = convert_currency(1.0, from_curr=data_currency, to_curr=target_currency)
curr_meta = get_currency_meta(target_currency)
curr_sym = curr_meta["symbol"]
curr_code = target_currency

if data_currency != target_currency:
    st.sidebar.info(f"Live Exchange Rate: 1 {data_currency} = {conv_rate:.4f} {target_currency}")

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
    budget = st.sidebar.slider(f"Marketing Budget ({curr_sym})", min_value=1000, max_value=50000, value=10000, step=1000)
    discount_rate = st.sidebar.slider("Discount Value (%)", min_value=5, max_value=30, value=15, step=1) / 100.0
    nuisance_choice = st.sidebar.selectbox("First-Stage Estimator", ["random_forest", "linear"], index=0)
else:
    sample_size = st.sidebar.slider("Store-Day Records", min_value=2000, max_value=15000, value=5000, step=1000)
    budget = st.sidebar.slider(f"Promotion Budget ({curr_sym})", min_value=5000, max_value=100000, value=25000, step=5000)
    discount_rate = st.sidebar.slider("Promo Cost Rate (%)", min_value=2, max_value=15, value=5, step=1) / 100.0
    nuisance_choice = st.sidebar.selectbox("First-Stage Estimator", ["linear", "random_forest"], index=0)


# Pipeline Execution with Caching
@st.cache_data(show_spinner="Computing causal effects and running optimization...")
def execute_causal_engine(dataset_choice: str, n_records: int, model_type: str, budget_val: float, disc_rate: float):
    if dataset_choice == "Synthetic Benchmark (Known Ground Truth)":
        df = generate_synthetic_data(n_samples=n_records, random_state=42)
        feat_cols = ["Age", "PriorSpend", "VisitFrequency", "LoyaltyScore"]
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
                archetypes.append("Persuadables (Targeted)")
            elif row["Sales"] > median_sales:
                archetypes.append("Sure Things (Excluded)")
            else:
                archetypes.append("Lost Causes (Excluded)")
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
                archetypes.append("Organic Flagship Volume (Excluded)")
            else:
                archetypes.append("Low Sensitivity Baseline (Excluded)")
        df["Archetype"] = archetypes
        
        return df, dml, naive_lift, knapsack_res, policy_df, feat_cols

# Convert user-selected budget from target currency to data currency for the solver
budget_in_data_currency = budget / conv_rate if conv_rate > 0 else budget

# Run Engine
data_df, model_dml, naive_lift_val, knapsack_out, policy_comp_df, active_feats = execute_causal_engine(
    dataset_type, sample_size, nuisance_choice, float(budget_in_data_currency), discount_rate
)

# Scale all financial figures by conv_rate into the target currency
converted_dml_ate = model_dml.ate_ * conv_rate
converted_naive_lift = naive_lift_val * conv_rate

# -------------------------------------------------------------
# 1. EXECUTIVE KPI BANNER
# -------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

budget_spent = knapsack_out["total_cost"] * conv_rate
total_incremental = knapsack_out["total_incremental_value"] * conv_rate
roi_val = knapsack_out["estimated_roi"]
targeted_count = knapsack_out["n_targeted"]
total_count = len(data_df)
skipped_count = total_count - targeted_count

with kpi_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Allocated Capital</div>
        <div class="kpi-value val-blue">{curr_sym}{budget_spent:,.0f}</div>
        <div class="kpi-subtext sub-muted">Limit: {curr_sym}{budget:,.0f} (100% adherence)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Incremental Revenue</div>
        <div class="kpi-value val-green">+{curr_sym}{total_incremental:,.0f}</div>
        <div class="kpi-subtext val-green">Causal revenue directly attributable to AI</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Campaign ROI</div>
        <div class="kpi-value val-purple">{roi_val:.1f}x</div>
        <div class="kpi-subtext val-purple">Return per allocated {curr_code}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Inefficient Spend Avoided</div>
        <div class="kpi-value val-amber">{skipped_count:,} Excluded</div>
        <div class="kpi-subtext val-amber">Margin preserved from organic buyers</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# 2. SIDE-BY-SIDE VISUAL PROOF (CONFOUNDING & STRATEGY TOURNAMENT)
# -------------------------------------------------------------
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown("""
    <div class="chart-box">
        <div class="chart-title">Confounding Breakdown: Correlation vs. True Causation</div>
        <div class="chart-desc">Standard analytics credits organic demand to promotions. Double ML isolates genuine incremental lift.</div>
    """, unsafe_allow_html=True)
    
    bias_val = max(converted_naive_lift - converted_dml_ate, 0.0)
    fig_trap = go.Figure()
    
    fig_trap.add_trace(go.Bar(
        name="True Causal Lift (DML)",
        x=["Observed Promotion Sales"],
        y=[converted_dml_ate],
        marker_color="#3B82F6",
        text=[f"+{curr_sym}{converted_dml_ate:,.0f}"],
        textposition="auto"
    ))
    fig_trap.add_trace(go.Bar(
        name="Confounding (Calendar / Store Characteristics)",
        x=["Observed Promotion Sales"],
        y=[bias_val],
        marker_color="#F59E0B",
        text=[f"Bias: -{curr_sym}{bias_val:,.0f}"],
        textposition="auto"
    ))
    fig_trap.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        height=310,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(gridcolor="#1E293B", title=f"Sales Value ({curr_sym})"),
        xaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_trap, use_container_width=True)
    
    pct_bias = (bias_val / max(converted_naive_lift, 1e-6)) * 100
    st.markdown(f"""
    <div class="insight-pill">
        <b>Finding:</b> {pct_bias:.0f}% of observed promotional lift corresponds to organic traffic that occurs independent of discounts.
    </div>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown(f"""
    <div class="chart-box">
        <div class="chart-title">Allocation Strategy Benchmark</div>
        <div class="chart-desc">Incremental revenue generated under identical {curr_sym}{budget:,.0f} promotional budget.</div>
    """, unsafe_allow_html=True)
    
    display_policy_df = policy_comp_df.copy()
    display_policy_df["Incremental_Gain"] = display_policy_df["Incremental_Gain"] * conv_rate
    display_policy_df["Budget_Spent"] = display_policy_df["Budget_Spent"] * conv_rate
    
    colors = ["#10B981", "#F59E0B", "#64748B"]
    fig_tourney = px.bar(
        display_policy_df,
        x="Policy",
        y="Incremental_Gain",
        color="Policy",
        color_discrete_sequence=colors,
        text_auto=".2s"
    )
    fig_tourney.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis=dict(gridcolor="#1E293B", title=f"Incremental Revenue ({curr_sym})"),
        xaxis=dict(showgrid=False, title="")
    )
    st.plotly_chart(fig_tourney, use_container_width=True)
    
    causal_gain = display_policy_df.iloc[0]["Incremental_Gain"]
    heuristic_gain = max(display_policy_df.iloc[1]["Incremental_Gain"], 1.0)
    multiple = causal_gain / heuristic_gain
    st.markdown(f"""
    <div class="insight-pill">
        <b>Efficiency Gain:</b> EconoCausal yields <b>{multiple:.1f}x higher incremental revenue</b> relative to standard top-spender allocation.
    </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. POPULATION SEGMENTATION & QINI CURVE
# -------------------------------------------------------------
row3_col1, row3_col2 = st.columns([1, 1])

with row3_col1:
    st.markdown("""
    <div class="chart-box">
        <div class="chart-title">Population Segmentation (Causal Archetypes)</div>
        <div class="chart-desc">Budget prioritization filters out sure things to concentrate funds on persuadables.</div>
    """, unsafe_allow_html=True)
    
    arch_counts = data_df["Archetype"].value_counts().reset_index()
    arch_counts.columns = ["Segment", "Count"]
    
    fig_donut = px.pie(
        arch_counts,
        names="Segment",
        values="Count",
        hole=0.55,
        color_discrete_sequence=["#3B82F6", "#F59E0B", "#475569"]
    )
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row3_col2:
    st.markdown("""
    <div class="chart-box">
        <div class="chart-title">Qini Uplift Curve (Cumulative Gain)</div>
        <div class="chart-desc">Higher curve trajectory demonstrates faster incremental revenue capture per candidate.</div>
    """, unsafe_allow_html=True)
    
    curve_data = compute_uplift_curve(
        data_df["Sales"].values,
        data_df["Discount"].values,
        data_df["Estimated_ITE"].values,
        n_bins=10
    )
    qini_score = compute_qini_score(curve_data)
    
    # Scale Qini curve lift by conversion rate
    scaled_qini = curve_data["CumulativeQini"] * conv_rate
    scaled_random = curve_data["RandomBaseline"] * conv_rate
    
    fig_qini = go.Figure()
    fig_qini.add_trace(go.Scatter(
        x=curve_data["FractionTargeted"] * 100,
        y=scaled_qini,
        mode="lines+markers",
        name="EconoCausal Policy",
        line=dict(color="#10B981", width=3)
    ))
    fig_qini.add_trace(go.Scatter(
        x=curve_data["FractionTargeted"] * 100,
        y=scaled_random,
        mode="lines",
        name="Random Allocation",
        line=dict(color="#EF4444", dash="dash")
    ))
    fig_qini.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E293B", title="% of Candidates Targeted"),
        yaxis=dict(gridcolor="#1E293B", title=f"Cumulative Incremental Lift ({curr_sym})")
    )
    st.plotly_chart(fig_qini, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. OPERATIONAL ACTION LIST & EXPORT
# -------------------------------------------------------------
st.markdown("""
<div class="chart-box">
    <div class="chart-title">Operational Allocation Schedule</div>
    <div class="chart-desc">Optimized customer and store decisions generated by the Knapsack solver.</div>
""", unsafe_allow_html=True)

action_cols = ["CustomerID", "Sales", "Estimated_ITE", "Archetype", "Decision"]
display_df = data_df[action_cols].copy()
display_df["Sales"] = np.round(display_df["Sales"] * conv_rate, 2)
display_df["Estimated_ITE"] = np.round(display_df["Estimated_ITE"] * conv_rate, 2)
display_df = display_df.sort_values(by="Estimated_ITE", ascending=False).reset_index(drop=True)

display_df.rename(columns={
    "CustomerID": "Candidate ID",
    "Sales": f"Historical Sales ({curr_sym})",
    "Estimated_ITE": f"Predicted Uplift ({curr_sym})",
    "Decision": "Allocation Decision"
}, inplace=True)

col_tbl, col_dl = st.columns([4, 1])
with col_tbl:
    st.dataframe(
        display_df.head(15).style.map(
            lambda v: "color: #34D399; font-weight: 600;" if v == "TARGET" else "color: #64748B;",
            subset=["Allocation Decision"]
        ),
        use_container_width=True,
        height=260
    )

with col_dl:
    st.write("")
    st.write("")
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Export Allocation Schedule ({curr_code} CSV)",
        data=csv_bytes,
        file_name=f"EconoCausal_Allocation_Schedule_{curr_code}.csv",
        mime="text/csv",
        help="Export complete allocation schedule for marketing and operational deployment.",
        use_container_width=True
    )
    st.markdown(f"""
    <div style="font-size: 12px; color: #94A3B8; margin-top: 10px;">
        <div><b>Targeted:</b> {targeted_count:,}</div>
        <div><b>Excluded:</b> {skipped_count:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
