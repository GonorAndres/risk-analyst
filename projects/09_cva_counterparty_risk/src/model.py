"""CVA model orchestrating simulation, exposure, and CVA computation.

Ties together Vasicek rate simulation, swap valuation, exposure
profiling, and CVA/DVA calculation into a single coherent pipeline.

Reference: Gregory (2020), Ch. 12-14 -- CVA, DVA, Wrong-Way Risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from credit import hazard_rate_from_cds, survival_probability, default_probability
from cva import compute_bilateral_cva, compute_cva, cva_by_netting_set
from exposure import apply_collateral, compute_exposure_profiles
from instruments import InterestRateSwap, simulate_rate_paths


class CVAModel:
    """End-to-end CVA computation model.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys matching default.yaml
        structure.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._rate_paths: NDArray[np.float64] | None = None
        self._times: NDArray[np.float64] | None = None
        self._mtm_values: NDArray[np.float64] | None = None
        self._profiles: dict | None = None

    def simulate(self) -> dict:
        """Simulate rate paths, compute swap values, compute exposure profiles.

        Returns
        -------
        dict
            Dictionary with keys 'rate_paths', 'times', 'mtm_values',
            'profiles'.
        """
        ir = self.config["interest_rate"]
        sw = self.config["swap"]
        sim = self.config["simulation"]
        seed = self.config["random_seed"]

        # Simulate Vasicek rate paths
        self._rate_paths, self._times = simulate_rate_paths(
            r0=ir["r0"],
            kappa=ir["kappa"],
            theta=ir["theta"],
            sigma=ir["sigma"],
            T=sw["tenor"],
            n_steps=sim["n_steps"],
            n_paths=sim["n_paths"],
            seed=seed,
        )

        # Create swap and compute MTM values
        swap = InterestRateSwap(
            notional=sw["notional"],
            fixed_rate=sw["fixed_rate"],
            tenor=sw["tenor"],
            payment_freq=sw["payment_freq"],
            seed=seed,
        )
        self._mtm_values = swap.simulate_values(self._rate_paths, self._times)

        # Compute exposure profiles
        self._profiles = compute_exposure_profiles(self._mtm_values, self._times)

        return {
            "rate_paths": self._rate_paths,
            "times": self._times,
            "mtm_values": self._mtm_values,
            "profiles": self._profiles,
        }

    def compute_cva(self) -> dict:
        """Compute CVA, DVA, and bilateral CVA.

        Returns
        -------
        dict
            Dictionary with keys 'cva', 'dva', 'bcva'.
        """
        if self._profiles is None:
            self.simulate()

        cpty = self.config["counterparty"]
        own = self.config["own_credit"]

        cpty_hazard = hazard_rate_from_cds(cpty["cds_spread"], cpty["recovery"])
        own_hazard = hazard_rate_from_cds(own["cds_spread"], own["recovery"])

        result = compute_bilateral_cva(
            ee=self._profiles["ee"],
            ene=self._profiles["ene"],
            times=self._times,
            cpty_hazard=cpty_hazard,
            own_hazard=own_hazard,
            recovery=cpty["recovery"],
        )

        return result

    def netting_analysis(self) -> pd.DataFrame:
        """Compare gross vs netted CVA for multiple trades.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: metric, value.
        """
        if self._rate_paths is None:
            self.simulate()

        sw = self.config["swap"]
        netting = self.config["netting"]
        cpty = self.config["counterparty"]
        seed = self.config["random_seed"]

        cpty_hazard = hazard_rate_from_cds(cpty["cds_spread"], cpty["recovery"])

        # Create multiple swaps with different fixed rates to get
        # diversified exposure (some positive, some negative)
        trades_mtm = []
        base_rate = sw["fixed_rate"]
        offsets = np.linspace(-0.005, 0.005, netting["n_trades"])

        for i, offset in enumerate(offsets):
            swap = InterestRateSwap(
                notional=sw["notional"],
                fixed_rate=base_rate + offset,
                tenor=sw["tenor"],
                payment_freq=sw["payment_freq"],
                seed=seed + i,
            )
            mtm = swap.simulate_values(self._rate_paths, self._times)
            trades_mtm.append(mtm)

        netting_result = cva_by_netting_set(
            trades_mtm=trades_mtm,
            times=self._times,
            hazard_rate=cpty_hazard,
            recovery=cpty["recovery"],
        )

        df = pd.DataFrame(
            {
                "metric": ["Gross CVA", "Net CVA", "Netting Benefit", "Benefit (%)"],
                "value": [
                    netting_result["gross_cva"],
                    netting_result["net_cva"],
                    netting_result["netting_benefit"],
                    netting_result["benefit_pct"],
                ],
            }
        )
        return df

    def wrong_way_risk(self, correlation: float) -> dict:
        """Introduce correlation between exposure and default intensity.

        Models wrong-way risk by correlating the exposure paths with
        the default intensity.  Positive correlation means higher
        exposure when default is more likely (wrong-way risk).

        Parameters
        ----------
        correlation : float
            Correlation between exposure and default intensity.
            Positive = wrong-way risk, negative = right-way risk.

        Returns
        -------
        dict
            Dictionary with keys 'cva', 'correlation'.
        """
        if self._mtm_values is None or self._times is None:
            self.simulate()

        cpty = self.config["counterparty"]
        cpty_hazard = hazard_rate_from_cds(cpty["cds_spread"], cpty["recovery"])

        n_paths = self._mtm_values.shape[0]

        # Model wrong-way risk: adjust hazard rate based on exposure level
        # For each path, increase hazard rate proportional to exposure
        # h_adj(t) = h * exp(correlation * z(t))
        # where z(t) is a standardised measure of the exposure at time t
        positive_exp = np.maximum(self._mtm_values, 0.0)

        # Standardise exposure across paths at each time point
        exp_mean = np.mean(positive_exp, axis=0, keepdims=True)
        exp_std = np.std(positive_exp, axis=0, keepdims=True)
        # Avoid division by zero
        exp_std = np.where(exp_std < 1e-12, 1.0, exp_std)
        z_exposure = (positive_exp - exp_mean) / exp_std

        # Adjusted hazard rate per path per time
        h_adjusted = cpty_hazard * np.exp(correlation * z_exposure)

        # Compute CVA with path-dependent hazard rates
        lgd = 1.0 - cpty["recovery"]
        cva = 0.0

        for i in range(1, len(self._times)):
            dt = self._times[i] - self._times[i - 1]
            discount = np.exp(-0.03 * self._times[i])

            # Path-dependent marginal default probability
            marginal_pd = h_adjusted[:, i] * dt

            # CVA contribution: average of (EE * marginal_PD) across paths
            cva += lgd * float(np.mean(positive_exp[:, i] * marginal_pd)) * discount

        return {"cva": float(cva), "correlation": correlation}

    def exposure_summary(self) -> pd.DataFrame:
        """Create summary table of exposure and default probability metrics.

        Returns
        -------
        pd.DataFrame
            Table with columns: time, EE, PFE_975, PFE_99, PD.
        """
        if self._profiles is None:
            self.simulate()

        cpty = self.config["counterparty"]
        cpty_hazard = hazard_rate_from_cds(cpty["cds_spread"], cpty["recovery"])

        pd_values = [default_probability(cpty_hazard, t) for t in self._times]

        df = pd.DataFrame(
            {
                "time": self._times,
                "EE": self._profiles["ee"],
                "PFE_975": self._profiles["pfe_975"],
                "PFE_99": self._profiles["pfe_99"],
                "PD": pd_values,
            }
        )
        return df
