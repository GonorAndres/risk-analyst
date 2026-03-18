"""Data loading, synthetic generation, and preprocessing for credit scoring.

Provides a deterministic synthetic credit dataset with realistic feature
distributions and correlations, avoiding external download dependencies
while preserving the statistical properties needed for scorecard development.

The default rate is calibrated to ~5-10%, consistent with sub-prime consumer
lending portfolios (Siddiqi, 2017, Ch. 2).

References:
    - Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd ed., Ch. 2.
    - Thomas, L. C. (2009). *Consumer Credit Models*, Ch. 2: data preparation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_credit_data(
    n_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic credit data with realistic correlations and default rate.

    Features generated:
        - income: annual income ($20k-$200k, log-normal)
        - age: borrower age (21-70)
        - employment_length: years at current employer (0-30)
        - loan_amount: requested loan ($1k-$50k)
        - interest_rate: assigned APR (5%-30%)
        - dti: debt-to-income ratio (0-60%)
        - credit_history_length: years of credit history (0-40)
        - n_delinquencies: past delinquencies (0-10, zero-inflated Poisson)
        - loan_purpose: categorical (debt_consolidation, credit_card,
          home_improvement, major_purchase, other)

    Target:
        - default: binary (1 = default, 0 = current), ~5-10% default rate

    The default probability is modeled via a latent logistic function of
    the features, ensuring realistic monotonic relationships (higher DTI
    -> higher PD, higher income -> lower PD, etc.).

    Parameters
    ----------
    n_samples : int
        Number of observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic credit dataset with a ``default`` column.
    """
    rng = np.random.default_rng(seed)

    # -- Feature generation with realistic marginals --

    # Income: log-normal, median ~$55k
    income = rng.lognormal(mean=10.9, sigma=0.5, size=n_samples)
    income = np.clip(income, 20_000, 200_000)

    # Age: truncated normal centered at 40
    age = rng.normal(loc=40, scale=12, size=n_samples)
    age = np.clip(age, 21, 70).astype(int)

    # Employment length: correlated with age
    employment_length = rng.exponential(scale=5, size=n_samples)
    employment_length = np.clip(employment_length, 0, np.minimum(age - 18, 30))
    employment_length = np.round(employment_length, 1)

    # Loan amount: correlated with income
    loan_amount = income * rng.uniform(0.05, 0.5, size=n_samples)
    loan_amount = np.clip(loan_amount, 1_000, 50_000).astype(int)

    # DTI: debt-to-income ratio, higher for riskier borrowers
    dti = rng.beta(2, 5, size=n_samples) * 60
    dti = np.clip(dti, 0, 60)

    # Credit history length: correlated with age
    credit_history_length = (age - 18) * rng.uniform(0.2, 0.9, size=n_samples)
    credit_history_length = np.clip(credit_history_length, 0, 40).astype(int)

    # Number of delinquencies: zero-inflated Poisson
    has_delinquency = rng.binomial(1, 0.3, size=n_samples)
    n_delinquencies = has_delinquency * rng.poisson(lam=1.5, size=n_samples)
    n_delinquencies = np.clip(n_delinquencies, 0, 10)

    # Interest rate: higher for riskier profiles (correlated with dti and delinquencies)
    base_rate = 5.0 + dti * 0.2 + n_delinquencies * 1.5
    interest_rate = base_rate + rng.normal(0, 2, size=n_samples)
    interest_rate = np.clip(interest_rate, 5.0, 30.0)

    # Loan purpose: categorical
    purposes = ["debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "other"]
    purpose_probs = [0.35, 0.25, 0.15, 0.15, 0.10]
    loan_purpose = rng.choice(purposes, size=n_samples, p=purpose_probs)

    # Purpose risk offset (debt consolidation and credit card are riskier)
    purpose_risk = np.where(
        loan_purpose == "debt_consolidation", 0.3,
        np.where(
            loan_purpose == "credit_card", 0.2,
            np.where(
                loan_purpose == "other", 0.1,
                0.0,
            ),
        ),
    )

    # -- Default generation via latent logistic model --
    # Standardize features for the latent model
    def _standardize(x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    logit = (
        -3.2                                        # intercept (controls base rate ~7%)
        - 0.6 * _standardize(income)                # higher income -> lower PD
        - 0.3 * _standardize(np.float64(age))       # older -> lower PD
        - 0.2 * _standardize(employment_length)     # longer employment -> lower PD
        + 0.3 * _standardize(np.float64(loan_amount))  # bigger loan -> higher PD
        + 0.5 * _standardize(interest_rate)         # higher rate -> higher PD
        + 0.7 * _standardize(dti)                   # higher DTI -> higher PD
        - 0.3 * _standardize(np.float64(credit_history_length))
        + 0.8 * _standardize(np.float64(n_delinquencies))
        + purpose_risk                              # purpose effect
        + rng.normal(0, 0.5, size=n_samples)        # idiosyncratic noise
    )

    prob_default = 1.0 / (1.0 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    # -- Create a synthetic "application_date" for temporal splitting --
    # Spread applications over 2 years
    days_offset = rng.integers(0, 730, size=n_samples)
    days_offset = np.sort(days_offset)  # sorted for temporal ordering
    base_date = pd.Timestamp("2022-01-01")
    application_date = pd.to_datetime(
        [base_date + pd.Timedelta(days=int(d)) for d in days_offset]
    )

    df = pd.DataFrame({
        "application_date": application_date,
        "income": np.round(income, 2),
        "age": age,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "interest_rate": np.round(interest_rate, 2),
        "dti": np.round(dti, 2),
        "credit_history_length": credit_history_length,
        "n_delinquencies": n_delinquencies,
        "loan_purpose": loan_purpose,
        "default": default,
    })

    return df


def load_credit_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load a credit risk dataset.

    If *path* is provided, reads a CSV from disk. Otherwise, generates a
    synthetic dataset using :func:`generate_synthetic_credit_data`.

    Parameters
    ----------
    path : str, Path, or None
        Path to a CSV file. ``None`` triggers synthetic data generation.

    Returns
    -------
    pd.DataFrame
        Credit dataset with features and a ``default`` target column.
    """
    if path is not None:
        return pd.read_csv(path, parse_dates=["application_date"])
    return generate_synthetic_credit_data()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning, missing value handling, and type casting.

    Steps:
        1. Drop rows with all-NaN features.
        2. Fill numeric NaNs with column medians.
        3. One-hot encode ``loan_purpose`` (if present).
        4. Cast integer-like columns to int.
        5. Ensure ``default`` is int (0/1).

    Parameters
    ----------
    df : pd.DataFrame
        Raw credit dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for modeling.
    """
    df = df.copy()

    # Drop fully-empty rows (excluding date and target)
    feature_cols = [c for c in df.columns if c not in ("application_date", "default")]
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # One-hot encode loan_purpose
    if "loan_purpose" in df.columns:
        dummies = pd.get_dummies(df["loan_purpose"], prefix="purpose", dtype=int)
        df = pd.concat([df.drop(columns=["loan_purpose"]), dummies], axis=1)

    # Ensure target is integer
    if "default" in df.columns:
        df["default"] = df["default"].astype(int)

    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split to avoid data leakage.

    Splits on ``application_date`` so that the test set contains the most
    recent observations (out-of-time validation, OOT).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with an ``application_date`` column.
    test_ratio : float
        Fraction of the data to reserve for testing.

    Returns
    -------
    train_df : pd.DataFrame
        Earlier observations for training.
    test_df : pd.DataFrame
        Later observations for testing (OOT).
    """
    df = df.sort_values("application_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df
