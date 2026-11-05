import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Set consistent, modern styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def plot_qini_curve(curve_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots Qini / Uplift curve comparing DML model targeting against random selection."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    ax.plot(
        curve_df["FractionTargeted"] * 100,
        curve_df["CumulativeQini"],
        label="EconoCausal DML Policy",
        color="#2b5c8f",
        linewidth=2.5
    )
    ax.plot(
        curve_df["FractionTargeted"] * 100,
        curve_df["RandomBaseline"],
        label="Random Targeting Baseline",
        color="#d95f02",
        linestyle="--",
        linewidth=1.8
    )
    
    ax.fill_between(
        curve_df["FractionTargeted"] * 100,
        curve_df["RandomBaseline"],
        curve_df["CumulativeQini"],
        color="#2b5c8f",
        alpha=0.15,
        label="Incremental Qini Gain"
    )
    
    ax.set_title("Uplift / Qini Curve: Cumulative Incremental Sales", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Percentage of Customers Targeted (%)", fontsize=11)
    ax.set_ylabel("Cumulative Incremental Sales ($)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def plot_ite_distribution(ite_preds: np.ndarray, save_path: Optional[Path] = None):
    """Plots histogram and KDE of estimated Individual Treatment Effects (ITE)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    sns.histplot(ite_preds, kde=True, color="#1b9e77", bins=35, ax=ax, alpha=0.6)
    mean_val = float(np.mean(ite_preds))
    ax.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean ATE = ${mean_val:.2f}")
    
    ax.set_title("Distribution of Individual Treatment Effects (ITE)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Estimated Incremental Sales per Discount ($)", fontsize=11)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def plot_cumulative_gains(curve_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots cumulative gain chart."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(curve_df["FractionTargeted"] * 100, curve_df["CumulativeQini"], color="#7570b3", linewidth=2.5)
    ax.set_title("Cumulative Gains from Targeted Discounts", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Fraction of Customers Discounted (%)", fontsize=11)
    ax.set_ylabel("Incremental Sales Captured", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def plot_budget_allocation_summary(comparison_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Bar chart comparing policies by Incremental Gain and ROI."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, comparison_df["Incremental_Gain"], width, label="Incremental Gain ($)", color="#2b5c8f")
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, comparison_df["ROI"], width, label="ROI (x)", color="#e7298a")
    
    ax1.set_ylabel("Incremental Gain ($)", color="#2b5c8f", fontweight="bold")
    ax2.set_ylabel("ROI (Multiplier)", color="#e7298a", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["Policy"])
    ax1.set_title("Policy Comparison: Causal DML vs Traditional Marketing", fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
