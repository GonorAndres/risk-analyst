# 12 -- Climate Risk Scenarios

Climate transition and physical risk quantification aligned with NGFS pathways and TCFD reporting.

## Objectives

- Map the six NGFS climate scenarios to sector-level financial impacts.
- Quantify transition risk via carbon intensity and stranded-asset exposure.
- Model physical risk under temperature and sea-level-rise projections.
- Produce TCFD-aligned scenario analysis reports with global sensitivity analysis (Sobol indices).

## Key Techniques

- NGFS Phase V integrated assessment model (IAM) scenario consumption
- Sector-level carbon intensity mapping and emission-factor decomposition
- Transition risk transmission: carbon price shock to sectoral equity repricing
- Physical risk modeling: damage functions, flood/heat stress overlays
- Sobol sensitivity analysis (first-order, total-order indices) via OpenTURNS
- Discounted cash flow adjustments under climate scenarios
- TCFD metrics: Scope 1/2/3 emissions, weighted average carbon intensity (WACI)

## Data Sources

- **NGFS Scenarios** -- Phase V scenario variables (orderly, disorderly, hot house)
- **NOAA / Climate Central** -- physical risk data (temperature anomalies, sea-level projections)

## Dependencies

```
pip install "risk-analyst[risk,ml]"
```

## References

1. Network for Greening the Financial System (2025). *NGFS Climate Scenarios -- Phase V*.
2. Task Force on Climate-related Financial Disclosures (2017). *Recommendations of the TCFD*.
3. Battiston, S., Mandel, A., Monasterolo, I., Schuetze, F., & Visentin, G. (2017). A climate stress-test of the financial system. *Nature Climate Change*, 7, 283--288.
