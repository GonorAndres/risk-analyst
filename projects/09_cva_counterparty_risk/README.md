# 09 -- CVA and Counterparty Credit Risk

Credit Valuation Adjustment calculation engine with exposure simulation and wrong-way risk.

## Objectives

- Bootstrap hazard rates and survival probabilities from CDS spread term structures.
- Simulate future portfolio exposures along interest rate and FX paths.
- Compute Expected Exposure (EE), Potential Future Exposure (PFE), and unilateral CVA.
- Model netting, collateral (CSA), and wrong-way risk effects on counterparty exposure.

## Key Techniques

- Hazard rate bootstrapping from CDS par spreads (piecewise constant intensity)
- Interest rate simulation (Hull-White, G2++) and FX path generation
- Exposure aggregation: EE, EPE, PFE profiles at portfolio and netting-set level
- CVA as discounted expected loss over the exposure profile
- Netting benefit quantification and CSA collateral margining
- Wrong-way risk modeling via copula-based default-exposure correlation
- Bilateral CVA (DVA) and Funding Valuation Adjustment (FVA) extensions

## Data Sources

- **Simulated curves** -- interest rate and FX scenarios
- **QuantLib** -- yield curve bootstrapping and CDS pricing

## Dependencies

```
pip install "risk-analyst[risk]"
```

## References

1. Gregory, J. (2020). *The xVA Challenge: Counterparty Risk, Funding, Collateral, Capital and Initial Margin*. 4th ed. Wiley.
2. Pykhtin, M. & Zhu, S. (2007). A guide to modelling counterparty credit risk. *GARP Risk Review*, July/August.
3. Brigo, D., Morini, M., & Pallavicini, A. (2013). *Counterparty Credit Risk, Collateral and Funding*. Wiley.
