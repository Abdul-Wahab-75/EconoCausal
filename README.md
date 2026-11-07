# EconoCausal
## Dynamic Pricing via Double Machine Learning

**Domain:** Economics & Causal AI  
**Project:** Project 2 — Infotact Solutions Advanced Data Science & ML Engineering

## 1. Project Overview

EconoCausal is a causal machine learning and budget optimization project designed to support smarter discount allocation.

Traditional predictive models identify customers who may purchase or churn, but they do not necessarily estimate whether a discount causes a change in customer behaviour.

EconoCausal uses Double Machine Learning (DML) to estimate treatment effects and a budget-constrained optimization approach to select customers for discount targeting.

**Main objective:** Estimate the potential incremental effect of discounts and allocate a limited marketing budget using model-based customer values.

## 2. Problem Statement

Businesses can waste marketing budgets by offering discounts to customers who would have purchased without an incentive.

The project aims to distinguish customers based on their estimated response to a discount and compare causal targeting with alternative allocation strategies.

The assignment describes the following customer categories:

- **Persuadables:** Customers who respond positively to discounts.
- **Sure Things:** Customers who may purchase without a discount.
- **Lost Causes:** Customers who are unlikely to purchase even with a discount.
- **Sleeping Dogs:** Customers whose response to a discount may be negative.

These categories describe the business concept. They should not be interpreted as individually verified customer labels unless the implementation explicitly calculates them.

## 3. Project Workflow

The implemented project follows this general workflow:

1. Load and prepare the dataset.
2. Generate treatment-effect estimates using the DML modelling pipeline.
3. Evaluate uplift and causal targeting performance.
4. Allocate a fixed budget using a 0-1 Knapsack optimization approach.
5. Compare Causal DML, High-Spender and Random targeting policies.
6. Display results through an interactive dashboard.
7. Expose selected model operations through a REST API.
8. Detect numerical feature distribution changes using data drift checks.
9. Generate natural-language summaries of allocation strategies.

## 4. Assignment Development Plan

The following four-week plan is taken from the EconoCausal assignment in the Infotact project document.

### Week 1 — Causal Graphing and Dashboard Foundation

**Causal ML and Optimization**
- Define a causal Directed Acyclic Graph (DAG).
- Identify confounders, treatment and outcome variables.
- Prepare a mock retail dataset.

**Visualization and UI**
- Create the dashboard foundation.
- Provide a way to load historical campaign data.
- Provide budget input and configuration.

### Week 2 — Double Machine Learning and Uplift Evaluation

**Causal ML and Optimization**
- Train Double Machine Learning models.
- Estimate Individual Treatment Effects (ITE).
- Perform causal validation and audit.

**Visualization and UI**
- Display Qini and uplift evaluation charts.
- Support filtering and visualization of customer treatment-effect estimates.

### Week 3 — Prescriptive Optimization

**Causal ML and Optimization**
- Implement budget-constrained optimization using SciPy.
- Allocate discounts using estimated treatment effects.
- Compare allocation strategies.

**Visualization and UI**
- Display the customer allocation matrix.
- Show selected customers and their allocation information.

### Week 4 — API, Data Drift and Dashboard Refinement

**Causal ML and Optimization**
- Expose the causal inference engine through a REST API.
- Implement an automated data drift detector.

**Visualization and UI**
- Refine the dashboard.
- Add natural-language strategy summaries.

**Final Review:** Present the integrated causal and prescriptive analytics solution.

## 5. Implemented Optimization Approach

The project uses a 0-1 Knapsack optimization approach implemented with SciPy.

Three allocation strategies are compared:

1. **Causal DML Optimal:** Selects customers using estimated treatment effects and budget-constrained optimization.
2. **High-Spender Heuristic:** Prioritizes customers with higher historical `PriorSpend`; if that field is unavailable, estimated prices are used.
3. **Random Targeting:** Selects customers in a reproducible random order.

All three strategies use the same estimated customer discount costs and budget limit.

### Optimization formulation

Maximize:

`Σ (v_i × t_i)`

Subject to:

`Σ (c_i × t_i) ≤ Budget`

`t_i ∈ {0, 1}`

Where:

- `t_i` indicates whether customer `i` is selected.
- `v_i` is the estimated incremental value.
- `c_i` is the estimated discount cost.
- `Budget` is the available marketing budget.

### Value calculation in the current implementation

For Rossmann revenue ITE:

`v_i = max(ITE_i × 0.10, 0)`

For the Synthetic Benchmark:

`v_i = max(ITE_i × item_price_i × 0.10, 0)`

Estimated discount cost:

`c_i = max(item_price_i × discount_rate, 0.50)`

The 10% factor and minimum discount cost of 0.50 are modelling assumptions, not verified actual business margins or discount costs.

The optimizer uses SciPy's Mixed-Integer Linear Programming (MILP) solver with a greedy fallback when the MILP solver is unsuccessful.

## 6. Evaluation Metrics

The project uses the following evaluation and allocation metrics:

- **Qini Curve:** Evaluates cumulative uplift from treatment-effect-based targeting. It is not the optimizer's revenue estimate.
- **Estimated Incremental Value:** Sum of estimated values for selected customers.
- **Estimated ROI:** Estimated incremental value divided by estimated discount cost.
- **Budget Utilization:** Percentage of the available budget used.
- **Lift vs Random:** Percentage difference in estimated incremental gain compared with random targeting.

All optimizer values are model-based estimates. They do not guarantee realized revenue, profit or business outcomes. A controlled experiment is needed to validate actual business impact.

## 7. Technology Stack

| Area | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Causal Modelling | Double Machine Learning implementation |
| Optimization | SciPy MILP / HiGHS |
| Evaluation | Qini and uplift metrics |
| Dashboard | Streamlit |
| REST API | FastAPI |
| API Server | Uvicorn |
| Data Drift | PSI and Kolmogorov-Smirnov test |
| Testing | Pytest |

**Implementation note:** The assignment describes a React and Plotly experimentation UI. The current repository dashboard is implemented using Streamlit. This is an implementation difference and is documented transparently rather than claiming that a React frontend was built.

## 8. Project Structure

The following are the main implementation paths confirmed in the project:

```text
EconoCausal/
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── models/
│   ├── optimization/
│   │   ├── knapsack.py
│   │   └── policy_allocator.py
│   ├── evaluation/
│   │   └── drift.py
│   ├── dashboard/
│   │   ├── app.py
│   │   └── strategy_summary.py
│   └── utils/
│
├── tests/
│   ├── test_api.py
│   ├── test_drift.py
│   └── test_strategy_summary.py
│
├── reports/
│   ├── final_report.md
│   └── EconoCausal_Final_Presentation.pptx
│
├── requirements.txt
└── README.md
```

This is a high-level project structure. Other source files and test files may also be present.

## 9. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Abdul-Wahab-75/EconoCausal.git
cd EconoCausal
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If the project package installation is required:

```bash
python -m pip install -e .
```

## 10. Run the Dashboard

From the project root:

```bash
streamlit run src/dashboard/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser.

## 11. REST API

The project provides a REST API for interacting with the DML inference engine.

Start the API:

```bash
python -m uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health and model status |
| POST | `/fit` | Fit the DML model |
| POST | `/predict` | Predict treatment effects |
| GET | `/model-info` | Retrieve fitted model information |
| POST | `/drift` | Check numerical feature distribution drift |

The fitted API model is held in application memory. Restarting the API clears the fitted model state, so the model must be fitted again before prediction.

## 12. Data Drift Detection

The drift module compares reference and current numerical feature distributions.

It uses:

- **Population Stability Index (PSI):** Measures distribution changes.
- **Kolmogorov-Smirnov (KS) Test:** Tests whether two numerical samples have statistically different distributions.

A feature is flagged when its PSI reaches the configured threshold or its KS p-value falls below the configured significance level.

Drift detection indicates distribution changes. It does not independently prove model degradation or a change in causal effects.

## 13. Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run the API and drift tests separately:

```bash
python -m pytest tests/test_api.py tests/test_drift.py -v
```

Tests are intended to check model, optimization, dashboard-supporting logic, API and data drift functionality.

## 14. Limitations and Modelling Assumptions

- Causal estimates depend on the available data, model assumptions and observed confounders.
- Observational estimates do not automatically establish causal validity.
- The optimization uses assumed margin and minimum discount-cost values where actual business costs are unavailable.
- Estimated ROI is not realized financial ROI.
- Data drift does not by itself demonstrate model performance degradation.
- The current API model state is in memory and must be fitted again after an API restart.
- The dashboard uses Streamlit rather than the React and Plotly UI described in the assignment.

## 15. Final Project Objective

EconoCausal aims to demonstrate how causal machine learning and budget-constrained optimization can support more informed discount allocation.

The final solution combines treatment-effect estimation, policy comparison, optimization, evaluation, dashboard visualization, API access and data drift monitoring.

**Repository:** https://github.com/Abdul-Wahab-75/EconoCausal