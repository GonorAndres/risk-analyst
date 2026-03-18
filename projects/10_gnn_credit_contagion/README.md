# 10 -- GNN-Based Credit Contagion

Graph neural networks for modeling systemic risk and default contagion in financial networks.

## Objectives

- Construct interbank and supply-chain financial networks from synthetic data.
- Train GNN models (GCN, GAT, GraphSAGE) for node-level default probability prediction.
- Simulate cascade and contagion dynamics through the network.
- Compute systemic risk metrics (CoVaR, SRISK, DebtRank) and identify systemically important nodes.

## Key Techniques

- Graph construction: interbank lending, CDS cross-holdings, supply-chain linkages
- Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)
- GraphSAGE for inductive learning on unseen nodes
- Temporal graph networks for time-evolving financial networks
- DebtRank and cascade simulation algorithms
- CoVaR estimation via quantile regression on graph embeddings
- SRISK (Systemic Risk Index) as a node-level capital shortfall measure

## Data Sources

- **Synthetic financial networks** -- generated with configurable topology (Erdos-Renyi, scale-free, core-periphery)

## Dependencies

```
pip install "risk-analyst[ml]"
```

## References

1. Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). DebtRank: too central to fail? Financial networks, the FED and systemic risk. *Scientific Reports*, 2, 541.
2. Guo, K. et al. (2025). Credit risk contagion modeling using graph neural networks. *ACM BAIDE 2025*.
3. Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
