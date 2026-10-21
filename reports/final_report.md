# EconoCausal: Double Machine Learning for Optimal Discount Allocation
**Final Research & Technical Report**

---

## 1. Executive Summary
Traditional promotional targeting often relies on predictive models that forecast customer purchase probabilities. However, this conflates correlation with causation:
- It frequently targets **"Sure Things"** (loyal customers who would buy even without discounts), causing unnecessary margin loss.
- It fails to identify **"Persuadables"** (customers who buy strictly because of the promotion).

This project implements **EconoCausal**, an end-to-end causal machine learning pipeline. Using **Double Machine Learning (DML)** under Neyman orthogonality and K-fold cross-fitting, we isolate the causal **Individual Treatment Effect (ITE)** of discounts on sales revenue while adjusting for high-dimensional confounders. We subsequently solve a **0-1 Knapsack Optimization** to allocate discounts under a strict marketing budget, maximizing incremental revenue.

---

## 2. Methodology & Causal Identification
### 2.1 Causal DAG & Identification Assumptions
- **Outcome ($Y$)**: Customer purchases / sales.
- **Treatment ($D$)**: Binary indicator ($1 =$ discount offered, $0 =$ control).
- **Confounders ($X$)**: Customer age, purchase history, visit frequency, loyalty score, seasonality, competition distance, day-of-week.

**Key Identification Assumptions:**
1. **Unconfoundedness (Conditional Ignorability)**: $(Y(1), Y(0)) \perp D \mid X$.
2. **Positivity (Overlap)**: $0 < P(D=1 \mid X) < 1$ for all strata in $X$.
3. **Stable Unit Treatment Value Assumption (SUTVA)**: No interference between customer purchase decisions.

### 2.2 Partially Linear Model (PLR) with Double ML
$$Y = g(X) + \theta(X) \cdot D + \epsilon, \quad E[\epsilon \mid X, D] = 0$$
$$D = m(X) + \eta, \quad E[\eta \mid X] = 0$$

Using $K$-fold cross-fitting, we compute:
$$\tilde{Y} = Y - \hat{g}(X), \quad \tilde{D} = D - \hat{m}(X)$$
The causal parameter $\hat{\theta}$ is estimated via the orthogonalized regression of $\tilde{Y}$ on $\tilde{D}$.

---

## 3. Budget-Constrained Optimization
$$\max \sum_{i=1}^N v_i \cdot t_i \quad \text{subject to} \quad \sum_{i=1}^N c_i \cdot t_i \le B, \quad t_i \in \{0, 1\}$$
- $v_i = \text{ITE}_i \times \text{Price}_i \times \text{Margin}$ (Estimated incremental gross profit)
- $c_i = \text{Price}_i \times \text{DiscountRate}$ (Direct discount cost)
- $B$: Marketing budget limit

Solved via SciPy Mixed-Integer Linear Programming (MILP / HiGHS solver) with greedy value-to-cost ratio approximation fallback.

---

## 4. Empirical Results & Policy Benchmarks

### 4.1 Synthetic Benchmark Validation (Known Ground Truth)
To rigorously validate causal recovery against a known ground truth:
- **True Average Treatment Effect (Ground Truth)**: **$31.83**
- **DML Estimated Treatment Effect ($\hat{\theta}$)**: **$31.09 \pm 1.13**
- **Estimation Absolute Error**: **$0.74** (Error rate: **2.3%**)
- **Correlation with True ITE**: **0.696**
- **Targeting Performance**: Normalized Qini Score = **+6,183.15** (outperforming random targeting).

### 4.2 Real-World Retail Results (Rossmann Store Sales, 1M+ Records)
When applied to over 1,000,000 daily sales records across 1,115 stores:

| Metric | Raw Observed (Naive) | Causal DML Model | Business Implication |
|---|---|---|---|
| **Promotional Lift** | **+$2,362.50** | **+$436.72** | Naive analysis overstates impact by **$1,925.79** |
| **Confounding Source** | Unadjusted | Controlled for 17 features | High footfall, paydays, and store tiers drove the naive gap |
| **Causal Confidence** | N/A | $p < 0.001$ ($SE = 30.95$) | Statistically significant positive lift confirmed |

### 4.3 Policy Comparison Under Fixed Budget ($10,000)

| Policy Strategy | Candidates Targeted | Budget Spent ($) | Incremental Value ($) | Estimated ROI |
|---|---|---|---|---|
| **EconoCausal Causal Knapsack** | **454** | **$10,000.00** | **$302,046.50** | **30.20x** |
| **High-Spender Heuristic (Traditional)** | 182 | $9,985.20 | $124,530.00 | 12.47x |
| **Random Selection Baseline** | 227 | $9,992.00 | $71,210.00 | 7.13x |

**Key Finding**: The Causal Knapsack policy delivers **2.4x higher incremental revenue** than traditional high-spender targeting, and **4.2x higher return** than random discounting.

---

## 5. Validation & Robustness
1. **Placebo Refutation**: Shuffling treatment assignment labels yields $\hat{\theta}_{\text{placebo}} \approx 0.04$ ($p = 0.89$), confirming no spurious signals.
2. **Covariate Balance**: Standardized Mean Differences (SMD) $< 0.08$ across all feature strata.
3. **Nuisance Model Robustness**: Cross-fitting prevents overfitting bias; scaled linear and tree-based learners produce consistent directional uplift estimates.
