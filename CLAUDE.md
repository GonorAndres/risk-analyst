# Risk Analyst -- Frontier Quantitative Risk Analysis

## Project Vision

A comprehensive, progressive curriculum of **12+ hands-on Python projects** implementing frontier quantitative risk analysis techniques. Each project combines rigorous mathematical theory (LaTeX-rendered) with production-quality code, serving as both a **learning resource** and a **portfolio showcase** for versatile quant roles (risk analyst, quant researcher, actuarial).

**Author:** Andres Gonzalez Ortega -- UNAM Actuarial Science graduate. Unique positioning at the intersection of actuarial mathematics, financial engineering, and machine learning.

---

## Repository Structure

```
risk-analyst/
├── CLAUDE.md                          # This file -- project instructions for Claude
├── README.md                          # Repo overview, project catalog, quickstart
├── pyproject.toml                     # Project metadata, dependencies
├── requirements.txt                   # Pinned dependencies
│
├── docs/                              # LaTeX theory documents
│   ├── 00_mathematical_foundations/   # Shared math prerequisites
│   └── XX_project_name/              # Per-project theory PDFs + .tex sources
│
├── src/                               # Shared library code
│   └── risk_analyst/
│       ├── __init__.py
│       ├── data/                      # Data ingestion and preprocessing
│       ├── models/                    # Reusable model implementations
│       ├── measures/                  # Risk measure computations
│       ├── simulation/                # Monte Carlo engines
│       ├── visualization/             # Plotting utilities
│       └── utils/                     # Helpers (config, logging, validation)
│
├── projects/                          # Self-contained project modules
│   ├── 01_portfolio_risk_dashboard/
│   ├── 02_monte_carlo_engine/
│   ├── 03_credit_scoring_ml/
│   ├── 04_volatility_modeling/
│   ├── 05_evt_tail_risk/
│   ├── 06_copula_dependency/
│   ├── 07_stress_testing_framework/
│   ├── 08_deep_hedging/
│   ├── 09_cva_counterparty_risk/
│   ├── 10_gnn_credit_contagion/
│   ├── 11_conformal_risk_prediction/
│   ├── 12_climate_risk_scenarios/
│   └── 13_rl_portfolio_risk/
│
├── notebooks/                         # Exploratory Jupyter notebooks (drafts only)
├── tests/                             # pytest suite mirroring src/ structure
├── data/                              # Local data cache (gitignored, reproducible)
├── subagents_outputs/                 # Claude research outputs (reference only)
└── .github/
    └── workflows/                     # CI: lint, type-check, test
```

Each project directory follows this internal structure:
```
projects/XX_project_name/
├── README.md              # Project overview, objectives, key results
├── theory.tex             # LaTeX document with mathematical foundations
├── theory.pdf             # Compiled PDF
├── src/                   # Project-specific code
│   ├── __init__.py
│   ├── model.py           # Core model implementation
│   ├── data.py            # Data loading and preprocessing
│   └── evaluate.py        # Backtesting, validation, metrics
├── notebooks/
│   └── walkthrough.ipynb  # Step-by-step guided notebook
├── tests/
│   └── test_model.py      # Unit tests
├── configs/
│   └── default.yaml       # Model parameters (externalized, never hardcoded)
└── results/               # Outputs: figures, tables, reports
```

---

## Project Curriculum (Progressive Difficulty)

### Tier 1: Foundations

| # | Project | Key Techniques | Risk Measures |
|---|---------|----------------|---------------|
| 01 | **Portfolio Risk Dashboard** | Historical/parametric/MC VaR, rolling analysis, backtesting (Kupiec, Christoffersen, traffic-light) | VaR, CVaR, Maximum Drawdown, Sharpe, Sortino |
| 02 | **Monte Carlo Simulation Engine** | GBM, Cholesky correlation, variance reduction (antithetic, control variates, importance sampling), option pricing | Portfolio VaR/CVaR from simulated P&L |
| 03 | **Credit Scoring with ML** | Logistic regression, XGBoost/LightGBM, SHAP/LIME explainability, WoE/IV, calibration (Platt/isotonic) | PD, AUC, KS, Gini, Brier score |
| 04 | **Volatility Modeling** | GARCH, GJR-GARCH, EGARCH, APARCH, HAR-RV, regime-switching (Markov HMM) | Conditional VaR, volatility term structure |

### Tier 2: Intermediate-Advanced

| # | Project | Key Techniques | Risk Measures |
|---|---------|----------------|---------------|
| 05 | **EVT for Tail Risk** | Block maxima (GEV), peaks-over-threshold (GPD), threshold selection, non-stationary EVT | EVT-based VaR/ES at 99.5%/99.9% |
| 06 | **Copula Dependency Modeling** | Gaussian, t, Clayton, Gumbel, Frank copulas; vine copulas (R-vine); GARCH-filtered marginals | Tail dependence, joint VaR under different copula assumptions |
| 07 | **Stress Testing Framework** | Fed DFAST scenarios, macro-to-portfolio transmission, migration matrices, reverse stress testing | Stressed capital ratios, break-point scenarios |

### Tier 3: Frontier / Research-Grade

| # | Project | Key Techniques | Risk Measures |
|---|---------|----------------|---------------|
| 08 | **Deep Hedging** | Neural network hedging strategies, model-free pricing, transaction cost optimization | Hedging P&L distribution, CVaR of hedging error |
| 09 | **CVA Counterparty Risk** | Hazard rate bootstrapping, exposure simulation, netting/collateral, wrong-way risk | CVA, EE, PFE profiles |
| 10 | **GNN Credit Contagion** | Graph neural networks on interbank/supply-chain networks, temporal graph learning, cascade modeling | Systemic risk metrics (CoVaR, SRISK) |
| 11 | **Conformal Risk Prediction** | Conformal prediction for distribution-free risk bounds, conformal risk control | Coverage-guaranteed VaR/ES intervals |
| 12 | **Climate Risk Scenarios** | NGFS pathways, transition risk (carbon pricing), physical risk, TCFD reporting | Climate VaR, stranded asset exposure |
| 13 | **RL for Portfolio Risk** | Reinforcement learning agents optimizing risk-adjusted returns, CVaR-constrained MDPs | Dynamic risk allocation, tail risk parity |

---

## Technology Stack

### Core
- **Python 3.11+** (single language, no R/Julia)
- **numpy, pandas, scipy** -- numerical foundation
- **matplotlib, plotly** -- static and interactive visualization

### Risk & Finance
- **QuantLib-Python** -- derivatives pricing, yield curves, Greeks, CVA
- **riskfolio-lib** -- portfolio optimization (24 risk measures)
- **arch** -- GARCH family volatility modeling
- **pyextremes** -- extreme value analysis (block maxima, POT)
- **copulae / pycop** -- copula modeling
- **chainladder** -- actuarial reserving (for actuarial bridge projects)
- **PyPortfolioOpt** -- portfolio optimization (lighter alternative)
- **financepy** -- fixed income, credit, FX derivatives

### Machine Learning
- **scikit-learn** -- classical ML, preprocessing, evaluation
- **xgboost / lightgbm** -- gradient boosting for credit scoring
- **PyTorch** -- deep learning (deep hedging, GNNs, neural SDEs)
- **PyTorch Geometric** -- graph neural networks
- **shap** -- explainability (SHAP values)
- **numpyro** -- Bayesian inference (JAX backend, fast MCMC)
- **stable-baselines3** -- reinforcement learning

### Data
- **yfinance** -- market data
- **fredapi** -- FRED macroeconomic data
- **pandas-datareader** -- multiple data sources

### Engineering
- **pytest** -- testing
- **mypy** -- type checking
- **ruff** -- linting and formatting
- **pydantic** -- configuration validation
- **streamlit** -- interactive dashboards
- **LaTeX (texlive)** -- theory document compilation

---

## Coding Standards

### General Principles
- **Separation of concerns:** Data ingestion, model logic, evaluation, and visualization live in separate modules.
- **Configuration externalized:** All model parameters (confidence levels, window sizes, thresholds) in YAML configs via pydantic. Never hardcode.
- **Reproducibility:** Pin random seeds, pin dependencies, document data sources. An uninvolved third party should reproduce results (SR 11-7 standard).
- **Type hints everywhere:** All function signatures typed. Enforce with mypy in CI.
- **No over-engineering:** Write the minimum code needed. Three similar lines > premature abstraction.

### Naming Conventions
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Mathematical variables in code should match their LaTeX notation where practical (e.g., `sigma` not `vol`, `lambda_t` not `hazard_rate`)

### Testing
- Every model gets unit tests covering:
  - Known analytic solutions (e.g., Black-Scholes closed-form)
  - Boundary conditions (zero vol, zero time, extreme params)
  - Monotonicity properties (VaR increases with confidence level)
  - Convergence of Monte Carlo estimates
- Use `pytest.approx()` for numerical comparisons, never exact equality on floats
- Property-based testing with `hypothesis` for invariant checking
- Backtesting: Kupiec, Christoffersen, traffic-light for all VaR models

### Documentation
- Each project gets a **LaTeX theory document** (`theory.tex` -> `theory.pdf`) with:
  - Mathematical problem statement and motivation
  - Full derivations of key results
  - Algorithm pseudocode
  - Connection to the codebase (which function implements which equation)
  - References to original papers
- Code docstrings reference the specific equation number in the theory doc
- Jupyter walkthrough notebook per project for guided exploration

### Git Conventions
- Branch per project: `project/XX-project-name`
- Commit messages: imperative mood, reference project number (e.g., `[P03] add SHAP feature importance visualization`)
- PRs require: passing tests, type checks, linting

---

## Data Sources

| Source | Content | Access |
|--------|---------|--------|
| **yfinance** | Equity/ETF/index prices, options chains | `pip install yfinance` |
| **FRED** | 800k+ macro time series (GDP, rates, spreads, unemployment) | `pip install fredapi` + API key |
| **Kenneth French** | Fama-French factors (3F, 5F, momentum) | Free download |
| **AQR** | Factor returns for research | Free download |
| **Lending Club** | 2.9M+ loans for credit scoring | Kaggle |
| **Home Credit** | Alternative credit data with bureau records | Kaggle |
| **Fed DFAST** | Stress test scenarios (baseline/adverse/severe) | federalreserve.gov |
| **EBA** | EU-wide stress test results (64 banks) | eba.europa.eu |
| **NGFS** | 6 climate pathways with macro-financial projections to 2100 | ngfs.net |
| **CBOE** | VIX, volatility indices | cboe.com |

Data is **never committed** to the repo. Each project includes a `data.py` module that downloads and caches data locally with reproducible scripts.

---

## Key Theoretical References

### Foundational Texts
- McNeil, Frey & Embrechts -- *Quantitative Risk Management* (QRM bible)
- Glasserman -- *Monte Carlo Methods in Financial Engineering*
- Hull -- *Options, Futures, and Other Derivatives*
- Nelsen -- *An Introduction to Copulas*
- de Haan & Ferreira -- *Extreme Value Theory: An Introduction*

### Frontier Papers (2024--2026)
- Basel FRTB: ES at 97.5% replacing VaR at 99%
- Rough Bergomi: neural net calibration in <1s (Journal of FinTech, 2025)
- Neural SDEs for option pricing (Quantitative Finance, 2026)
- Conformal Risk Control (JMLR, 2024) + Conformal Risk Training (NeurIPS, 2025)
- GNN credit contagion on supply chains (ACM BAIDE, 2025)
- Deep hedging extensions with tail risk (2025)
- ICVaR-DRL for portfolio risk (2025)
- Robust Bernoulli mixture models for portfolio credit risk (arXiv, 2024)
- FABRICS framework for cyber risk quantification (Computers & Security, 2026)
- NGFS Phase V short-term scenarios (May 2025)

---

## Claude Instructions

### When building projects
1. Always start by reading the project's theory.tex (or relevant research in subagents_outputs/) to understand the mathematical foundations before writing code.
2. Implement models as clean Python classes with typed interfaces.
3. Write tests alongside the implementation, not after.
4. Use the shared `src/risk_analyst/` library for reusable components (simulation engines, risk measures, data loaders).
5. Every numerical result should be reproducible with a fixed random seed.
6. Externalize all parameters to YAML configs.
7. Create a Jupyter walkthrough that tells a story: problem -> theory -> implementation -> results -> interpretation.

### When writing theory documents
1. Use the LaTeX template in `docs/` (will be created).
2. Start with intuition and motivation before formal definitions.
3. Include full derivations -- don't skip steps. This is for learning.
4. Connect every theorem/algorithm to its code implementation with explicit cross-references.
5. Include numerical examples that match the code outputs.

### When adding dependencies
1. Add to `pyproject.toml` under the appropriate optional group.
2. Update `requirements.txt` with pinned versions.
3. Document why the dependency is needed in a code comment at the import site if it's not obvious.

### What NOT to do
- Do not hardcode file paths, API keys, or model parameters.
- Do not commit data files or large outputs to git.
- Do not skip mathematical rigor in theory docs for brevity.
- Do not add features beyond what the current project scope requires.
- Do not use R, Julia, or any language other than Python.
