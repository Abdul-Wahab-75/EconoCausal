# EconoCausal
## Dynamic Pricing via Double Machine Learning

**Project:** Project 2 — Infotact Solutions, Advanced Data Science & ML Engineering  
**Domain:** Economics & Causal AI  
**Repository:** https://github.com/Abdul-Wahab-75/EconoCausal  
**Contribution branch:** `prince-yadav`

## 1. Project Overview

EconoCausal is a causal machine-learning and prescriptive analytics platform for evaluating promotional strategies and allocating a limited discount budget.

Traditional predictive models estimate who is likely to purchase. Causal machine learning instead estimates how an outcome may change under a treatment, such as receiving a promotion. EconoCausal uses Double Machine Learning (DML) to estimate Individual Treatment Effects (ITE), evaluates uplift, compares targeting policies, and uses constrained optimization to select records under a budget.

The platform combines a Python causal modelling and optimization backend, a FastAPI REST API, a React dashboard with Plotly visualizations, Propensity Score Matching (PSM), DoWhy causal analysis, data-drift checks, and natural-language strategy summaries.

> **Important data note:** The Rossmann dataset contains store-day observations, not individual customer records, and does not provide actual item prices for this implementation. Therefore, results on this dataset are record-level estimates. Where a price is required, the dashboard uses an illustrative assumed price of ₹50 per record. These outputs are not verified customer-specific discount recommendations or actual realized revenue.

## 2. Problem Statement

Businesses can waste promotional spending by discounting customers who would have purchased without an incentive. This project explores how estimated treatment effects can help identify records that may respond to promotions and compare causal targeting with alternative allocation strategies.

The assignment describes four conceptual customer groups:
- **Persuadables:** May respond positively to a discount.
- **Sure Things:** May purchase without a discount.
- **Lost Causes:** May remain unlikely to purchase even with a discount.
- **Sleeping Dogs:** May respond negatively to a discount.

These are business concepts, not individually verified labels unless a specific implementation calculates them.

## 3. End-to-End Workflow

1. Load and prepare retail data.
2. Select treatment, outcome, and covariate columns.
3. Train the DML model and estimate ITE.
4. Review model statistics and treatment-effect estimates.
5. Evaluate uplift/Qini performance against a random baseline.
6. Allocate a fixed budget using SciPy optimization.
7. Compare Causal DML Optimal, High-Spender Heuristic, and Random Targeting.
8. Run propensity score matching and inspect covariate balance.
9. Run DoWhy causal analysis and available refutation checks.
10. Detect numerical feature distribution drift.
11. Generate a natural-language strategy summary.
12. Explore results in the React + Plotly dashboard or through the REST API.

## 4. Development Plan and Delivered Work

### Week 1 — Causal Graphing and Dashboard Foundation
- Established the causal workflow and identified treatment, outcome, and confounder roles.
- Prepared mock/synthetic retail data and integrated the Rossmann Store Sales dataset.
- Set up the dashboard foundation for CSV upload, model configuration, and budget input.

### Week 2 — Double Machine Learning and Uplift Evaluation
- Implemented the DML modelling workflow and ITE prediction.
- Added model evaluation outputs, including ATE and standard error.
- Added Qini/uplift evaluation with a random baseline.
- Added DoWhy causal analysis and refutation checks.
- Added PSM to compare treated and control records and inspect covariate balance.

### Week 3 — Prescriptive Optimization
- Implemented budget-constrained 0-1 Knapsack optimization using SciPy MILP, with a fallback approach.
- Added allocation outputs such as selected records, estimated costs, budget usage, incremental value, and estimated ROI.
- Added comparison of Causal DML Optimal, High-Spender Heuristic, and Random Targeting.
- Integrated allocation results into the dashboard.

### Week 4 — API, Data Drift, and Dashboard Refinement
- Exposed model, prediction, optimization, causal-analysis, matching, drift, uplift, and strategy-summary functionality through FastAPI.
- Implemented numerical data-drift checks using PSI and the Kolmogorov–Smirnov test.
- Added natural-language strategy summaries and policy comparison.
- Integrated the React frontend with the API and added Plotly-based result visualizations.
- Verified Python syntax, ran the automated test suite, and built the frontend for production.

## 5. Main Features

### 5.1 Double Machine Learning and ITE
The DML workflow estimates treatment effects while accounting for selected observed covariates. The dashboard displays model results and individual treatment-effect estimates.

### 5.2 Uplift and Qini Evaluation
The evaluation view plots model uplift/Qini performance against a random baseline. These metrics assess targeting/ranking performance; they are not realized revenue.

### 5.3 Propensity Score Matching
The PSM workflow matches treated and control observations using propensity scores and reports matching/balance information. Balance diagnostics should be reviewed before interpreting matched comparisons.

### 5.4 DoWhy Causal Analysis
The DoWhy workflow defines a causal model, estimates an effect where the selected variables permit it, and runs available refutation checks. An estimand that cannot be identified from the supplied variables may be unavailable; do not interpret that as a successful estimate.

### 5.5 SciPy Budget Optimization
The optimizer selects records subject to the configured budget and reports selected records, estimated spend, remaining budget, estimated incremental value, and estimated ROI.

The policy comparison includes:
1. **Causal DML Optimal**
2. **High-Spender Heuristic**
3. **Random Targeting**

### 5.6 Data Drift Detection
The drift module compares reference and current numerical feature distributions using:
- **Population Stability Index (PSI)**
- **Kolmogorov–Smirnov (KS) test**

A drift flag indicates a distribution change under configured thresholds. It does not by itself prove model degradation or a change in causal effects.

### 5.7 Natural-Language Strategy Summary
The strategy-summary API creates a readable overview of allocation, budget usage, estimated value/ROI, and policy comparison. This is decision support, not a guarantee of business outcomes.

### 5.8 React + Plotly Dashboard
The frontend provides CSV upload, treatment/outcome/feature configuration, model training, ITE and uplift views, budget optimization, causal diagnostics, drift checks, and strategy summaries.

## 6. Optimization Approach and Assumptions

The allocation problem is formulated as a 0-1 selection problem.

Maximize:

`Σ (v_i × x_i)`

Subject to:

`Σ (c_i × x_i) ≤ Budget`

`x_i ∈ {0, 1}`

Where:
- `x_i` indicates whether record `i` is selected.
- `v_i` is the model-based estimated incremental value.
- `c_i` is the estimated discount cost.
- `Budget` is the available marketing budget.

The implementation uses SciPy's MILP solver and a fallback method if the MILP solver cannot provide a solution.

**Assumptions and interpretation**
- The Rossmann sample does not contain actual item prices; ₹50 per record is an illustrative assumption in the dashboard.
- Configured modelling factors and minimum estimated costs are assumptions, not verified margins or actual campaign costs.
- Estimated incremental value and ROI are model outputs, not realized financial results.
- Observational data and model assumptions limit causal interpretation. A controlled experiment is needed to validate real-world impact.

## 7. Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Causal ML | EconML / Double Machine Learning |
| Causal analysis | DoWhy |
| Matching | Propensity Score Matching |
| Optimization | SciPy MILP |
| Evaluation | Qini and uplift metrics |
| Backend API | FastAPI, Uvicorn |
| Drift detection | PSI, Kolmogorov–Smirnov test |
| Frontend | React, Vite, JavaScript |
| Visualization | Plotly |
| Testing | Pytest |

## 8. Project Structure

High-level structure (some supporting modules and files are omitted):

```text
EconoCausal-Group/
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── api.js
│   ├── package.json
│   └── index.html
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
│   └── dashboard/
│       ├── app.py
│       └── strategy_summary.py
├── tests/
├── data/
├── requirements.txt
└── README.md
```

## 9. Installation

### 9.1 Clone the repository

```bash
git clone https://github.com/Abdul-Wahab-75/EconoCausal.git
cd EconoCausal
```

### 9.2 Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 9.3 Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

If the repository's packaging configuration is present and required:

```bash
python -m pip install -e .
```

### 9.4 Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## 10. Run the Application

Run the backend and frontend in separate terminals.

### Terminal 1 — FastAPI backend

From the project root, with the virtual environment activated:

```bash
python -m uvicorn src.api.main:app --reload
```

API base URL: `http://127.0.0.1:8000`  
Interactive API documentation: `http://127.0.0.1:8000/docs`

### Terminal 2 — React frontend

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`).

### Production frontend build

```bash
cd frontend
npm run build
```

The production build output is generated in `frontend/dist/`.

## 11. REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health check |
| POST | `/fit` | Fit the DML model |
| POST | `/predict` | Predict treatment effects |
| GET | `/model-info` | Retrieve fitted model information |
| POST | `/drift` | Detect numerical feature drift |
| POST | `/uplift` | Calculate uplift/Qini evaluation |
| POST | `/optimize` | Optimize allocation under a budget |
| POST | `/psm` | Run propensity score matching |
| POST | `/dowhy` | Run DoWhy causal analysis/refutation checks |
| POST | `/strategy-summary` | Generate strategy summary and policy comparison |

Use `/docs` to inspect request schemas and try endpoints interactively.

**Model lifecycle:** The fitted API model is held in application memory. Restarting the API clears the fitted model state; fit the model again before requesting predictions.

## 12. Testing and Validation

Run the complete test suite:

```bash
python -m pytest tests/ -q
```

Latest reported local validation for this contribution:
- Python compilation: `python -m py_compile src/api/main.py` — passed.
- Backend tests: **39 passed**.
- Frontend production build: `npm run build` — passed.
- The frontend build reported a large JavaScript bundle warning. The build nevertheless completed successfully; bundle splitting can be considered as a future performance improvement.

Third-party dependency warnings appeared during testing but did not cause test failures.

## 13. Limitations and Future Validation

- The Rossmann dataset represents store-day observations, not individual customers.
- Actual product/item prices and actual discount costs are not supplied by the Rossmann data used here; illustrative assumptions are used where needed.
- Record-level optimization results must not be presented as verified customer-specific pricing recommendations.
- DML estimates depend on treatment definition, covariate selection, model specification, data quality, and causal assumptions.
- Observational analysis does not replace a randomized controlled experiment.
- PSI/KS alerts identify distribution shifts; they do not independently prove loss of predictive accuracy.
- The API stores the fitted model in memory; persistent model storage is not implemented in this workflow.
- The production build reports a large frontend bundle size; this may be optimized later.

## 14. Final Summary

EconoCausal brings together Double Machine Learning, treatment-effect estimation, uplift evaluation, propensity score matching, DoWhy causal analysis, SciPy budget optimization, policy comparison, data-drift monitoring, natural-language strategy summaries, and a React + Plotly dashboard.

The project demonstrates an end-to-end causal and prescriptive analytics workflow. Numerical outputs should be interpreted in light of the dataset limitations and modelling assumptions described above.

---

**Repository:** https://github.com/Abdul-Wahab-75/EconoCausal  
**Contribution branch:** `prince-yadav`