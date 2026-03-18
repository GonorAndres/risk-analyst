"""Data ingestion: market data, FRED macroeconomic series, and dataset loaders."""

from risk_analyst.data.market import compute_losses, compute_returns, fetch_prices

__all__ = ["fetch_prices", "compute_returns", "compute_losses"]
