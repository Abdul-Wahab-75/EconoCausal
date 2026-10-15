# EconoCausal: Presentation Deck Outline (10-12 Slides)

### Slide 1: Title & Team
- Title: **EconoCausal: Double Machine Learning for Optimal Discount Allocation**
- Subtitle: *Maximizing Promotional ROI Through Causal Uplift and Knapsack Optimization*
- Team Members & Specific Roles (Data Engineer, Causal Modeler, Optimization Analyst, Project Lead)

### Slide 2: The Core Problem & Motivation
- Business dilemma: Indiscriminate discounting burns margin on customers who would buy anyway.
- The 4 Customer Quadrants: "Sure Things", "Lost Causes", "Sleeping Dogs", and "Persuadables".
- Why correlation is not causation: Standard regression biases discount effectiveness.

### Slide 3: Causal Framework & DAG
- Directed Acyclic Graph (DAG): Features $X \to D$, $X \to Y$, and Causal Link $D \to Y$.
- Core Assumptions: Unconfoundedness, Overlap (Positivity), SUTVA.

### Slide 4: Double Machine Learning (DML) Architecture
- Why Double ML? Chernozhukov et al. (2018) Robinson transformation.
- Stage 1: Flexible ML nuisance predictors ($E[Y|X]$ and $E[D|X]$).
- Stage 2: Orthogonalized score avoiding regularization bias via $K$-fold cross-fitting.

### Slide 5: Data Pipeline & Confounder Inspection
- Dataset description: Features, preprocessing, scaling, handling selection bias.
- Demonstration of confounding in baseline distributions.

### Slide 6: Model Validation & Robustness Checks
- Placebo Treatment permutation test.
- Covariate balance metrics (SMD $< 0.1$).
- Confidence intervals on Average Treatment Effect (ATE).

### Slide 7: Heterogeneous Treatment Effects (ITE)
- Distribution histogram of individual customer treatment effects.
- Identification of customer segments most responsive to promotions.

### Slide 8: Budget-Constrained Optimization (0-1 Knapsack)
- Mathematical Formulation: Maximize $\sum v_i t_i$ subject to $\sum c_i t_i \le B$.
- Solver implementation: Exact Mixed-Integer Linear Programming (SciPy MILP / HiGHS).

### Slide 9: Policy Comparison & Business ROI
- Benchmarking Causal Knapsack vs Heuristic vs Random allocation.
- ROI multiplier and incremental revenue lift.

### Slide 10: Evaluation Curves (Qini & Cumulative Gains)
- Uplift curve visualization and Area Under the Qini Curve (AUUC).
- Evidence that top deciles capture the majority of incremental gain.

### Slide 11: Demo Dashboard Walkthrough
- Overview of interactive Streamlit application (`src/dashboard/app.py`).
- Real-time simulation of budget adjustments.

### Slide 12: Business Recommendations, Limitations & Future Work
- Dynamic treatment effects over multi-period horizons.
- Ethical checks to ensure fair promotional targeting across demographics.
- Q&A.
