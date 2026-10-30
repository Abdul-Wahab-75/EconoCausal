# EconoCausal: Double Machine Learning for Optimal Discount Allocation
**Final Research & Technical Report**

---

## 1. Executive Summary
Traditional promotional targeting often relies on predictive models that forecast customer purchase probabilities. However, this conflates correlation with causation:
- It frequently targets **"Sure Things"** (loyal customers who would buy even without discounts), causing unnecessary margin loss.
- It fails to identify **"Persuadables"** (customers who buy strictly because of the promotion).

This project implements **EconoCausal**, an end-to-end causal machine learning pipeline. Using **Double Machine Learning (DML)** with K-fold cross-fitting, the system estimates the Average Treatment Effect (ATE) of discounts on sales while accounting for observed confounders. It also produces approximate Individual Treatment Effect (ITE) estimates using an interaction-based regression model. These estimates are then used in a **0-1 Knapsack Optimization** framework to allocate discounts under a fixed marketing budget, maximizing estimated incremental value.

---

## 2. Methodology & Causal Identification

### 2.1 Causal Identification Framework

The project uses Double Machine Learning (DML) to estimate the effect of discount treatment on sales while accounting for observed confounding variables.

- **Outcome (Y):** Sales or purchase-related outcome.
- **Treatment (D):** Binary indicator representing whether a discount or promotion was applied.
- **Confounders (X):** Observed variables relevant to treatment assignment and outcome prediction.

For the synthetic benchmark, the features include Age, PriorSpend, VisitFrequency, and LoyaltyScore.

For the Rossmann dataset, the model uses 17 engineered features covering calendar information, customer counts, store characteristics, competition, promotions, and holidays.

**Key Identification Assumptions:**

1. **Unconfoundedness (Conditional Ignorability):** Potential outcomes are conditionally independent of treatment given the observed features.
2. **Positivity (Overlap):** Each relevant feature group has a non-zero probability of receiving either treatment condition.
3. **Stable Unit Treatment Value Assumption (SUTVA):** The treatment applied to one unit does not interfere with the outcome of another unit.

These are identification assumptions underlying the causal framework; they are not claimed to have been independently proven by the reported experiments.

### 2.2 Partially Linear Double Machine Learning (PLR)

The native implementation uses a partially linear DML approach with the following nuisance models:

\[
Y = g(X) + \theta D + \epsilon
\]

\[
D = m(X) + \eta
\]

Here, \(g(X)\) models the outcome conditional on observed features, \(m(X)\) models treatment assignment, and \(\theta\) represents the average treatment effect.

Using K-fold cross-fitting, the implementation calculates residuals:

\[
\tilde{Y} = Y - \hat{g}(X)
\]

\[
\tilde{D} = D - \hat{m}(X)
\]

The average treatment effect is estimated through a second-stage regression of the residualized outcome on the residualized treatment.

The implementation also fits an interaction-based regression model to obtain approximate heterogeneous treatment effect (ITE) estimates for individual records.

The estimator supports configurable nuisance models, cross-fitting folds, and backend selection. The reported experiments use the native backend.

---

## 3. Budget-Constrained Optimization

The project uses a 0-1 Knapsack optimization model to allocate discounts under a fixed marketing budget.

\[
\max \sum_{i=1}^{N} v_i t_i
\]

Subject to:

\[
\sum_{i=1}^{N} c_i t_i \leq B,\qquad t_i \in \{0,1\}
\]

The implementation calculates incremental value and discount cost as follows:

- **Incremental Value:** \(v_i = \max(ITE_i \times Price_i \times 0.10,\ 0)\)
- **Discount Cost:** \(c_i = \max(Price_i \times DiscountRate,\ 0.50)\)
- **Budget (B):** Maximum permitted total discount expenditure.
- **Selection Variable (t):** Binary variable indicating whether a candidate is selected.

The 10% factor is the fixed gross-margin assumption used by the current implementation. The minimum discount cost is set to $0.50 per candidate.

The primary optimization method is SciPy Mixed-Integer Linear Programming (MILP), using the HiGHS solver. If the MILP solver fails, the implementation falls back to a greedy value-to-cost ratio allocation method.

The estimated return on investment (ROI) is calculated as total estimated incremental value divided by total discount cost.

For policy comparison, the causal optimization policy is evaluated against a historical high-spender heuristic and a random targeting baseline under the same budget and cost assumptions.

## 4. Empirical Results & Policy Benchmarks

### 4.1 Synthetic Benchmark Validation (Known Ground Truth)

To validate causal recovery against a known ground truth, the EconoCausal framework was evaluated using the synthetic benchmark dataset.

- **True Average Treatment Effect (Ground Truth):** **$31.83**
- **DML Estimated Treatment Effect:** **$31.09**
- **Standard Error:** **1.13**
- **Estimation Absolute Error:** **$0.74** (Error rate: **2.3%**)
- **Correlation with True ITE:** **0.696**
- **Targeting Performance:** Raw Qini area = **6,184.36**, indicating performance above random targeting in the reproduced benchmark.

### 4.2 Real-World Retail Results (Rossmann Store Sales)

The EconoCausal framework was evaluated on the Rossmann Store Sales dataset. After preprocessing, filtering open stores with positive sales, and merging store metadata, the final dataset contained 844,338 daily sales records across 1,115 stores.

The Double Machine Learning (DML) model used 17 engineered features, with Discount as the treatment variable and Sales as the outcome variable.

| Metric | Raw Observed (Naive) | Causal DML Model | Business Implication |
|---|---:|---:|---|
| **Promotional Lift** | **+$2,298.91** | **+$677.72** | Observed promotional lift is higher than the estimated causal effect |
| **Dataset Size** | **844,338 rows** | **17 features** | Filtered open-store records with positive sales |
| **Standard Error** | N/A | **2.6846** | Estimated standard error of the DML ATE |
| **95% Confidence Interval** | N/A | **[$672.46, $682.98]** | Positive estimated causal effect |

**Model Configuration:**
- Treatment Variable: `Discount`
- Outcome Variable: `Sales`
- Nuisance Model: Linear
- DML Backend: Native
- Cross-fitting Folds: 3
- Random State: 42

The results indicate that the observed difference in sales between promotional and non-promotional days is larger than the estimated causal treatment effect. This highlights the importance of controlling for observed confounding variables when estimating promotional impact.

### 4.3 Policy Comparison Under Fixed Budget ($10,000) - Synthetic Benchmark

The three targeting policies were evaluated using the same synthetic benchmark configuration and a fixed budget of $10,000.

| Policy Strategy | Candidates Targeted | Budget Spent ($) | Incremental Value ($) | Estimated ROI |
|---|---:|---:|---:|---:|
| **EconoCausal Causal Knapsack** | **454** | **$10,000.00** | **$302,048.52** | **30.20x** |
| **High-Spender Heuristic (Traditional)** | 346 | $9,999.93 | $192,288.42 | 19.23x |
| **Random Selection Baseline** | 527 | $9,999.66 | $213,656.59 | 21.37x |

**Key Finding:** In this reproduced synthetic benchmark, the Causal Knapsack policy achieved approximately 1.57x the incremental value of high-spender targeting and 1.41x the incremental value of random targeting.

These results represent the synthetic benchmark configuration and should not be interpreted as measured Rossmann business outcomes.

## 5. Validation & Robustness

1. **Placebo Refutation:** The placebo test shuffled treatment assignment labels over 3 iterations. The mean placebo ATE was approximately -0.0317, with a standard deviation of 0.7262. The test passed the implementation's criterion of an absolute mean placebo ATE below 1.0. No p-value was calculated by this implementation.

2. **Covariate Balance:** The raw standardized mean differences (SMD) across the four synthetic benchmark features were:
   - Age: 0.1565
   - PriorSpend: 0.3443
   - VisitFrequency: 0.2671
   - LoyaltyScore: 0.1604

   The maximum observed absolute SMD was approximately 0.3443. These raw pre-adjustment values indicate that the treated and control groups were not initially balanced under the 0.10 threshold.

3. **Validation Scope:** The placebo and covariate balance results are based on the synthetic benchmark configuration. They should be interpreted as validation checks for that experiment and not as Rossmann dataset results.

## 6. Conclusion

This project developed an end-to-end framework for estimating the causal impact of discounts and supporting budget-constrained customer targeting. The implementation combines data preprocessing, feature engineering, Double Machine Learning (DML), approximate Individual Treatment Effect (ITE) estimation, and 0-1 Knapsack optimization.

On the reproduced synthetic benchmark, the native DML model estimated an ATE of 31.09, compared with a true ATE of 31.83. The estimated ITE correlation was 0.696. Under the fixed $10,000 synthetic budget, the Causal Knapsack policy generated an estimated incremental value of $302,048.52.

On the filtered Rossmann dataset of 844,338 records, the estimated discount ATE was 677.72, with a 95% confidence interval of [$672.46, $682.98]. These are model estimates and should be interpreted in light of the assumptions and limitations below.

Overall, the project demonstrates how causal estimation and constrained optimization can be integrated into a dynamic pricing decision-support workflow.

## 7. Limitations and Future Work

- **Causal assumptions:** The interpretation of DML estimates depends on assumptions such as no unobserved confounding, sufficient overlap, and appropriate model specification. These assumptions have not been independently established by this project.
- **Approximate ITE:** Individual treatment effects are generated through an interaction-based regression approach and should be treated as approximate estimates, not exact individual causal effects.
- **Synthetic benchmark:** Policy comparison and the reported $10,000 budget results are from the synthetic benchmark. They are not measured real-world Rossmann revenue or ROI.
- **Covariate balance:** The reported raw synthetic standardized mean differences exceed the 0.10 threshold for all four features. This indicates initial imbalance and should be considered when interpreting results.
- **Placebo validation:** The placebo check used three iterations and an implementation-specific absolute mean ATE threshold of 1.0. It is a limited diagnostic, not proof of causal identification.
- **Rossmann evaluation:** The Rossmann results are observational model estimates. Further validation, sensitivity analysis, and business-level testing would be needed before operational use.
- **Future work:** Further work may include additional robustness checks, improved ITE validation, alternative nuisance models, and evaluation using prospective business experiments.

## 8. References

1. Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68. https://doi.org/10.1111/ectj.12097
2. SciPy documentation. Mixed-integer linear programming with `scipy.optimize.milp`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html


