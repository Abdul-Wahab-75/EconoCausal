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
- **Confounders ($X$)**: Customer age, purchase history, visit frequency, loyalty score, seasonality.

**Key Identification Assumptions:**
1. **Unconfoundedness (Conditional Ignorability)**: $(Y(1), Y(0)) \perp D \mid X$.
2. **Positivity (Overlap)**: $0 < P(D=1 \mid X) < 1$ for all strata in $X$.
3. **Stable Unit Treatment Value Assumption (SUTVA)**: No interference between customers.

### 2.2 Partially Linear Model (PLR) with Double ML
$$Y = g(X) + \theta(X) \cdot D + \epsilon, \quad E[\epsilon \mid X, D] = 0$$
$$D = m(X) + \eta, \quad E[\eta \mid X] = 0$$

Using $K$-fold cross-fitting, we compute:
$$\tilde{Y} = Y - \hat{g}(X), \quad \tilde{D} = D - \hat{m}(X)$$
The causal parameter $\hat{\theta}$ is estimated via the orthogonalized regression of $\tilde{Y}$ on $\tilde{D}$.

---

## 3. Budget-Constrained Optimization
$$\max \sum_{i=1}^N v_i \cdot t_i \quad \text{subject to} \quad \sum_{i=1}^N c_i \cdot t_i \le B, \quad t_i \in \{0, 1\}$$
- $v_i = \text{ITE}_i \times \text{Price}_i$ (Estimated incremental value)
- $c_i = \text{Price}_i \times \text{DiscountRate}$ (Direct discount cost)
- $B$: Marketing budget limit

---

## 4. Empirical Results & Policy Benchmarks
*(To be populated with empirical runs from `notebooks/04_evaluation_and_insights.ipynb`)*

| Policy | Customers Targeted | Total Budget Spent ($) | Incremental Gain ($) | ROI |
|---|---|---|---|---|
| **Causal DML Knapsack** | *TBD* | *TBD* | *TBD* | *TBD* |
| **High-Spender Heuristic** | *TBD* | *TBD* | *TBD* | *TBD* |
| **Random Targeting** | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 5. Validation & Robustness
1. **Placebo Refutation**: Shuffling treatment labels yields $\hat{\theta}_{\text{placebo}} \approx 0$.
2. **Covariate Balance**: Standardized Mean Differences (SMD) $< 0.10$.
3. **Confidence Bounds**: Bootstrap standard errors confirming statistical significance.
