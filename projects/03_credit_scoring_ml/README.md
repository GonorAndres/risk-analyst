# 03 -- Credit Scoring with Machine Learning

Probability-of-default model combining classical scorecard techniques with modern ML and explainability.

## Objectives

- Build a logistic regression baseline following traditional scorecard methodology (WoE binning, IV selection).
- Train gradient-boosted models (XGBoost, LightGBM) and compare discriminatory power.
- Ensure model explainability via SHAP and LIME at global and local levels.
- Calibrate predicted probabilities (Platt scaling, isotonic regression) and document per SR 11-7.

## Key Techniques

- Weight of Evidence (WoE) transformation and Information Value (IV) screening
- Logistic regression with L1/L2 regularization
- XGBoost and LightGBM with Bayesian hyperparameter tuning
- SHAP (TreeExplainer) and LIME for post-hoc interpretability
- Platt scaling and isotonic regression for probability calibration
- KS statistic, Gini coefficient, and precision-recall analysis
- SR 11-7 model risk management documentation framework

## Data Sources

- **Lending Club** loan dataset (Kaggle)
- **FRED** -- macroeconomic features (unemployment, Fed Funds rate)

## Dependencies

```
pip install "risk-analyst[ml]"
```

## References

1. Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*. 2nd ed. Wiley.
2. Board of Governors of the Federal Reserve System (2011). *SR 11-7: Guidance on Model Risk Management*.
3. Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
