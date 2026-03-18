# Risk Analyst

**Frontier Quantitative Risk Analysis: 13 Progressive Python Projects**

A hands-on curriculum from portfolio VaR to deep hedging, combining rigorous mathematical theory with production-quality implementations.

---

## Projects

| # | Project | Tier | Key Techniques |
|---|---------|------|----------------|
| 01 | Portfolio Risk Dashboard | Foundation | VaR, CVaR, backtesting (Kupiec, Christoffersen) |
| 02 | Monte Carlo Simulation Engine | Foundation | GBM, Cholesky correlation, variance reduction |
| 03 | Credit Scoring with ML | Foundation | XGBoost, SHAP/LIME, calibration |
| 04 | Volatility Modeling | Foundation | GARCH family, regime-switching, HAR-RV |
| 05 | EVT for Tail Risk | Intermediate | GEV, GPD, peaks-over-threshold |
| 06 | Copula Dependency Modeling | Intermediate | Vine copulas, tail dependence, GARCH-filtered marginals |
| 07 | Stress Testing Framework | Intermediate | DFAST scenarios, reverse stress testing |
| 08 | Deep Hedging | Frontier | Neural network hedging, model-free pricing |
| 09 | CVA Counterparty Risk | Frontier | Exposure simulation, wrong-way risk |
| 10 | GNN Credit Contagion | Frontier | Graph neural networks, cascade modeling |
| 11 | Conformal Risk Prediction | Frontier | Distribution-free risk bounds |
| 12 | Climate Risk Scenarios | Frontier | NGFS pathways, TCFD reporting |
| 13 | RL for Portfolio Risk | Frontier | CVaR-constrained MDPs, dynamic allocation |

Each project includes:
- **LaTeX theory document** with full mathematical derivations
- **Python implementation** with typed interfaces
- **Jupyter walkthrough** notebook
- **Unit tests** with analytic benchmarks
- **YAML configuration** for all parameters

## Quickstart

```bash
# Clone
git clone https://github.com/andtega349/risk-analyst.git
cd risk-analyst

# Install (core only)
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Run tests
pytest
```

## Structure

```
src/risk_analyst/       # Shared library (measures, simulation, data, viz)
projects/XX_name/       # Self-contained project modules
docs/XX_name/           # LaTeX theory documents
tests/                  # Global test suite
```

## Tech Stack

**Core:** numpy, pandas, scipy, matplotlib, plotly
**Risk:** QuantLib, riskfolio-lib, arch, pyextremes, copulae
**ML:** PyTorch, scikit-learn, XGBoost, PyG, numpyro, stable-baselines3
**Engineering:** pytest, mypy, ruff, pydantic, streamlit

## Author

Andres Gonzalez Ortega -- UNAM Actuarial Science

## License

MIT
