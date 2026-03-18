"""Market data ingestion and return computation.

Provides utilities for downloading adjusted close prices via yfinance
and computing log/simple returns and portfolio losses.

References:
    - Jorion (2007), Ch. 5: VaR definitions and return conventions.
    - McNeil, Frey & Embrechts (2015), Ch. 2: basic risk factor changes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (e.g., ["SPY", "QQQ"]).
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str | None
        End date in YYYY-MM-DD format.  ``None`` fetches up to the
        latest available date.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ticker holding
        the adjusted close price.
    """
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna()
    prices.index.name = "date"
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: Literal["log", "simple"] = "log",
) -> pd.DataFrame:
    """Compute asset returns from a price panel.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, one column per asset.
    method : {"log", "simple"}
        ``"log"``  -> r_t = ln(P_t / P_{t-1})
        ``"simple"`` -> r_t = P_t / P_{t-1} - 1

    Returns
    -------
    pd.DataFrame
        Returns with the same column layout as *prices*, first row dropped.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method '{method}'; expected 'log' or 'simple'.")

    return returns.dropna()


def compute_losses(
    returns: pd.DataFrame,
    weights: np.ndarray | list[float],
) -> pd.Series:
    """Compute portfolio losses (negative of portfolio returns).

    Losses are defined as L_t = -r_{p,t} so that a positive value
    represents an adverse move.  This convention aligns with the VaR
    literature (Jorion, 2007).

    Parameters
    ----------
    returns : pd.DataFrame
        Asset-level returns (T x N).
    weights : array-like
        Portfolio weights summing to 1, length N.

    Returns
    -------
    pd.Series
        Portfolio losses indexed by date.
    """
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != returns.shape[1]:
        raise ValueError(
            f"Weight vector length ({len(w)}) does not match "
            f"number of assets ({returns.shape[1]})."
        )

    portfolio_returns = returns.values @ w
    losses = -portfolio_returns
    return pd.Series(losses, index=returns.index, name="portfolio_loss")
