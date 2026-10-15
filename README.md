# 📈 EconoCausal

### Smart Discount Targeting using Causal AI & Budget Optimization

---

## 📌 What Is This Project About?

Every business gives discounts to increase sales — but **most companies waste money** by giving discounts to the wrong people:

| Customer Type | What Happens | Result |
|---|---|---|
| 🟢 **Persuadables** | They **only buy when they get a discount** | ✅ Money well spent |
| 🔵 **Sure Things** | They would have bought **even without a discount** | ❌ Wasted profit |
| 🔴 **Lost Causes** | They **won't buy no matter what** | ❌ Wasted budget |
| 🟡 **Sleeping Dogs** | Discounts actually **annoy them and push them away** | ❌ Negative effect |

> **Our Goal:** Build a smart system that figures out **which customers are "Persuadables"** and only gives discounts to them — all within a fixed marketing budget.

---

## 🧠 How Does It Work? (Simple Explanation)

We use a 3-step pipeline:

### Step 1: Find the Real Effect of Discounts (Causal AI)

Regular AI just looks at patterns like *"people who got discounts bought more."* But that's misleading — maybe those people would have bought anyway!

We use a technique called **Double Machine Learning (DML)** which removes this bias:

1. **First**, train an ML model to predict: *"How much would this customer buy WITHOUT a discount?"*
2. **Then**, train another ML model to predict: *"How likely is it that this customer was given a discount?"*
3. **Finally**, subtract out both predictions — what's left is the **true causal effect** of the discount on each customer.

This gives us a score called **ITE (Individual Treatment Effect)** — basically:
> *"If we give this specific customer a discount, how much EXTRA will they spend because of it?"*

### Step 2: Decide Who Gets the Discount (Budget Optimization)

We have a limited marketing budget (say $5,000). We can't discount everyone. So we solve a mathematical puzzle called the **Knapsack Problem**:

- Each customer has a **value** (their ITE — how much extra they'd spend)
- Each customer has a **cost** (the discount amount we'd give them)
- We pick the best combination of customers that **maximizes total extra sales** without going over budget

Think of it like packing a suitcase — you want to fit the most valuable items without exceeding the weight limit.

### Step 3: Measure & Compare (Evaluation)

We compare our smart approach against two baselines:
- **Random Targeting** — give discounts to random people
- **High-Spender Heuristic** — traditional marketing approach: discount the biggest spenders

We measure success using:
- **Qini Curve** — shows how much extra revenue we capture as we target more customers
- **ROI** — how many dollars of extra sales we get for every dollar spent on discounts

---

## 🔬 Technical Details (For Those Who Want the Math)

<details>
<summary>Click to expand the mathematical framework</summary>

### Causal Model (Partially Linear Regression)

```
Y = g(X) + θ(X) · D + ε       (Outcome equation)
D = m(X) + η                    (Treatment equation)
```

Where:
- `Y` = Sales (outcome)
- `D` = Discount given or not (0 or 1)
- `X` = Customer features (age, spend history, visit frequency, loyalty score)
- `g(X)` = Expected sales without discount
- `m(X)` = Probability of getting a discount
- `θ(X)` = **The causal effect we want to estimate (ITE)**

### DML Cross-Fitting Procedure

1. Split data into K folds
2. For each fold, train nuisance models on the OTHER folds to predict `g(X)` and `m(X)`
3. Compute residuals: `Ỹ = Y - ĝ(X)` and `D̃ = D - m̂(X)`
4. Regress `Ỹ` on `D̃` to get the unbiased treatment effect `θ̂`

### Budget Optimization (0-1 Knapsack)

```
Maximize:    Σ (ITE_i × price_i) × t_i
Subject to:  Σ (price_i × discount_rate) × t_i ≤ Budget
             t_i ∈ {0, 1}
```

Solved using Mixed-Integer Linear Programming (SciPy MILP / HiGHS solver).

### Key Assumptions

1. **Unconfoundedness** — We observe all the factors that influence both discount assignment and purchasing
2. **Positivity** — Every customer has some chance of receiving (or not receiving) a discount
3. **SUTVA** — One customer's discount doesn't affect another customer's buying decision

</details>

---

## 👥 Team Members & Who Works on What

We are a team of **4 members**. Each person owns a specific part of the code, so we can all work simultaneously without conflicts:

| # | Role | What They Do | Code Ownership | Notebook |
|---|---|---|---|---|
| 1 | **Data Engineer** | Loads data, cleans it, creates useful features | `src/data/` | `01_eda_and_preprocessing.ipynb` |
| 2 | **Causal Modeler** | Builds the DML model, estimates treatment effects, runs validity checks | `src/models/` | `02_causal_dml_modeling.ipynb` |
| 3 | **Optimization Analyst** | Solves the budget allocation problem, compares targeting strategies | `src/optimization/` | `03_budget_optimization.ipynb` |
| 4 | **Project Lead & Evaluator** | Measures performance (Qini curves), builds the demo dashboard, writes the report | `src/evaluation/` + `src/dashboard/` | `04_evaluation_and_insights.ipynb` |

---

## 📁 Project Folder Structure

```
EconoCausal/
│
├── data/                          # All datasets go here
│   ├── raw/                       #   └── Original downloaded data
│   ├── processed/                 #   └── Cleaned & ready-to-use data
│   └── synthetic/                 #   └── Fake data we generate for testing
│
├── notebooks/                     # Jupyter notebooks (one per team member)
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_causal_dml_modeling.ipynb
│   ├── 03_budget_optimization.ipynb
│   └── 04_evaluation_and_insights.ipynb
│
├── src/                           # Main Python code (importable package)
│   ├── data/                      #   └── Data loading, cleaning, feature creation
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   ├── synthetic.py           #       └── Generates fake customer data with known answers
│   │   └── features.py
│   ├── models/                    #   └── Causal inference engine
│   │   ├── nuisance.py            #       └── First-stage ML predictors
│   │   ├── dml_estimator.py       #       └── Double ML estimator (the core!)
│   │   └── robustness.py          #       └── Placebo tests & balance checks
│   ├── optimization/              #   └── Budget allocation solver
│   │   ├── knapsack.py            #       └── 0-1 Knapsack MILP solver
│   │   └── policy_allocator.py    #       └── Compare Causal vs Heuristic vs Random
│   ├── evaluation/                #   └── Metrics & charts
│   │   ├── metrics.py             #       └── Qini score, uplift curves, AUUC
│   │   └── visualizations.py      #       └── Publication-quality plots
│   ├── dashboard/                 #   └── Interactive web demo
│   │   └── app.py                 #       └── Streamlit dashboard
│   └── utils/                     #   └── Shared helpers
│       ├── config.py              #       └── File paths & default settings
│       └── logger.py              #       └── Logging utility
│
├── models/                        # Saved trained models (.pkl files)
├── reports/                       # Final report, slides, and charts
│   ├── figures/
│   ├── final_report.md
│   └── presentation_outline.md
├── tests/                         # Automated tests to verify everything works
│
├── .github/                       # GitHub CI pipeline & PR template
├── requirements.txt               # Python package dependencies
├── environment.yml                # Conda environment file
├── setup.py                       # Makes 'src' installable as a package
└── README.md                      # ← You are here!
```

---

## 🗓️ Project Roadmap (12 Weeks)

```
PHASE 1: FOUNDATIONS (Weeks 1-2)
├── Study Double ML theory & causal inference basics
├── Draw the Causal DAG (cause-and-effect diagram)
└── Set up the Git repo, environment, and CI pipeline

PHASE 2: DATA (Weeks 3-4)
├── Download real datasets (Rossmann Store Sales / UCI Online Retail)
├── Build the synthetic data generator (fake data with known answers for testing)
├── Clean data, handle missing values, create features
└── Run Exploratory Data Analysis (EDA)

PHASE 3: CAUSAL MODEL (Weeks 5-6)
├── Train first-stage ML models (predict sales & discount probability)
├── Implement K-fold cross-fitting
└── Get initial treatment effect estimates (ATE & ITE)

PHASE 4: VALIDATION (Weeks 7-8)
├── Run Placebo Test (shuffle treatment → effect should be ~0)
├── Check Covariate Balance (treated vs control groups should look similar)
└── Tune hyperparameters for better estimates

PHASE 5: OPTIMIZATION (Weeks 9-10)
├── Build the Knapsack budget solver
├── Compare Causal Policy vs High-Spender Heuristic vs Random
└── Integrate the full pipeline: Data → Model → Optimizer

PHASE 6: DELIVERY (Weeks 11-12)
├── Generate Qini curves, ROI tables, and publication charts
├── Build the interactive Streamlit demo dashboard
├── Write the final report and prepare the presentation (12 slides)
└── Rehearse and present!
```

---

## 🚀 How to Get Started

### 1. Clone the Repository
```bash
git clone https://github.com/Abdul-Wahab-75/EconoCausal.git
cd EconoCausal
```

### 2. Set Up Your Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Run the Tests (Make Sure Everything Works)
```bash
pytest tests/ -v
```

### 4. Launch the Demo Dashboard
```bash
streamlit run src/dashboard/app.py
```

### 5. Open the Notebooks
```bash
jupyter lab
```

---

## 🤝 How We Work Together on GitHub

### Branch Naming
Each member creates their own branch to work on:
```
member1/data-pipeline
member2/dml-modeling
member3/knapsack-optimization
member4/evaluation-dashboard
```

### Daily Workflow
```bash
# 1. Get the latest code
git checkout main
git pull origin main

# 2. Switch to your branch
git checkout -b member1/data-pipeline

# 3. Make your changes in YOUR folder only
#    (Member 1 works in src/data/, Member 2 in src/models/, etc.)

# 4. Test your changes
pytest tests/ -v

# 5. Commit and push
git add .
git commit -m "feat(data): added customer segmentation features"
git push origin member1/data-pipeline

# 6. Open a Pull Request on GitHub to merge into main
```

### Golden Rules
- ✅ Only edit files in YOUR assigned folder
- ✅ Always run `pytest tests/` before pushing
- ✅ Use descriptive commit messages
- ❌ Never push large data files or model binaries (they're gitignored)
- ❌ Never push directly to `main` — always use Pull Requests

---

## 📦 Key Technologies Used

| Category | Tools |
|---|---|
| **Causal Inference** | EconML, DoubleML, DoWhy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Optimization** | SciPy (MILP/HiGHS), PuLP |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Testing** | Pytest |
| **CI/CD** | GitHub Actions |

---

## 📊 What Success Looks Like

By the end of this project, we will demonstrate:

1. **Causal DML targeting outperforms random targeting** — our Qini curve is significantly above the random baseline
2. **Budget efficiency** — our Knapsack allocation achieves higher ROI than the traditional "discount the big spenders" approach
3. **Robustness** — Placebo tests confirm our effect is real (not a statistical artifact)
4. **A working demo** — stakeholders can interactively adjust budgets and see results in real-time via the Streamlit dashboard

---

> **Repository:** [github.com/Abdul-Wahab-75/EconoCausal](https://github.com/Abdul-Wahab-75/EconoCausal)
