import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def compute_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    ite_preds: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Computes cumulative uplift and Qini curve points by ordering individuals by predicted ITE.
    """
    df = pd.DataFrame({
        "y": y_true,
        "t": treatment,
        "ite": ite_preds
    }).sort_values(by="ite", ascending=False).reset_index(drop=True)
    
    n = len(df)
    bin_size = n // n_bins
    records = []
    
    cum_treated_y = 0.0
    cum_treated_n = 0
    cum_control_y = 0.0
    cum_control_n = 0
    
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        sub = df.iloc[start:end]
        
        t_sub = sub[sub["t"] == 1]
        c_sub = sub[sub["t"] == 0]
        
        cum_treated_y += t_sub["y"].sum()
        cum_treated_n += len(t_sub)
        cum_control_y += c_sub["y"].sum()
        cum_control_n += len(c_sub)
        
        # Qini point: Y_treated - (N_treated / N_control) * Y_control
        scale = (cum_treated_n / (cum_control_n + 1e-6))
        qini_val = cum_treated_y - scale * cum_control_y
        
        fraction_targeted = end / n
        records.append({
            "FractionTargeted": fraction_targeted,
            "TreatedCustomers": cum_treated_n,
            "ControlCustomers": cum_control_n,
            "CumulativeQini": qini_val,
            "RandomBaseline": (fraction_targeted * (df[df["t"]==1]["y"].sum() - (df["t"].sum() / ((1-df["t"]).sum() + 1e-6)) * df[df["t"]==0]["y"].sum()))
        })
        
    return pd.DataFrame(records)

def compute_qini_score(curve_df: pd.DataFrame) -> float:
    """Calculates normalized Qini coefficient (Area between model curve and random baseline)."""
    fractions = curve_df["FractionTargeted"].values
    model_curve = curve_df["CumulativeQini"].values
    baseline = curve_df["RandomBaseline"].values
    
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area_model = trapz_func(model_curve, fractions)
    area_baseline = trapz_func(baseline, fractions)
    return float(area_model - area_baseline)

def summarize_causal_metrics(ite_preds: np.ndarray, true_ite: np.ndarray = None) -> Dict[str, Any]:
    """Generates numerical summary of treatment effects."""
    top_decile_threshold = np.percentile(ite_preds, 90)
    top_decile_mean = float(np.mean(ite_preds[ite_preds >= top_decile_threshold]))
    
    summary = {
        "Mean_Estimated_ITE": float(np.mean(ite_preds)),
        "Std_Estimated_ITE": float(np.std(ite_preds)),
        "Median_Estimated_ITE": float(np.median(ite_preds)),
        "Top_10_Percent_Mean_ITE": top_decile_mean,
        "Min_ITE": float(np.min(ite_preds)),
        "Max_ITE": float(np.max(ite_preds))
    }
    
    if true_ite is not None:
        rmse = float(np.sqrt(np.mean((ite_preds - true_ite) ** 2)))
        mae = float(np.mean(np.abs(ite_preds - true_ite)))
        corr = float(np.corrcoef(ite_preds, true_ite)[0, 1])
        summary["GroundTruth_RMSE"] = rmse
        summary["GroundTruth_MAE"] = mae
        summary["GroundTruth_Correlation"] = corr
        
    return summary
