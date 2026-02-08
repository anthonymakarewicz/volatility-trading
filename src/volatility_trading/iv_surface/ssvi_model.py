import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .base_xssvi import xSSVI


class SSVI(xSSVI):
    """
    SSVI (Surface SVI) fit:
    w(k,T) = θ(T)/2 (1 + ρ φ(θ) k + sqrt((φ(θ) k + ρ)^2 + 1 - ρ²))

    φ(θ) = η θ^{-γ}
    """

    def __init__(self, bounds=None, **kwargs):
        super().__init__(**kwargs)
        default_bounds = [
            (-0.999, 0.999),  # ρ
            (1e-3, 2.0),  # η
            (1e-3, 0.999),  # γ
        ]
        self.bounds = bounds or default_bounds

    @classmethod
    def _w(cls, k, theta, params):
        rho, eta, gamma = params
        phi = cls._phi(theta, eta, gamma)
        return (
            0.5
            * theta
            * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
        )

    def _objective(self, k, theta, w, weights, params):
        rho, eta, gamma = params
        w_model = self._w(k, theta, params)
        sse = np.sum(weights * (w_model - w) ** 2)

        # ---- soft no-arb penalties (butterfly + calendar) ----
        # Butterfly (wing) caps (sufficient):
        theta_min, theta_max = np.min(theta), np.max(theta)

        # θφ(θ)(1+|ρ|) < 4
        eta_cap_3 = 4.0 / ((1.0 + abs(rho)) * theta_max ** (1.0 - gamma))

        # θφ(θ)^2(1+|ρ|) ≤ 4
        eta_cap_4 = 2.0 / np.sqrt((1.0 + abs(rho)) * theta_min ** (1.0 - 2.0 * gamma))

        eta_cap = min(eta_cap_3, eta_cap_4)
        if self.butterfly_arb_strict and eta >= eta_cap:
            sse += 1e8 * (eta - eta_cap) ** 2  # hinge penalty

        return sse

    def get_params(self) -> dict:
        if not hasattr(self, "params"):
            raise RuntimeError("Model not fitted.")

        rho, eta, gamma = self.params
        spot = self._spot
        T_grid = self.theta_interp.x
        theta_grid = self.theta_interp(T_grid)  # recover values at knots

        return {
            "rho": float(rho),
            "eta": float(eta),
            "gamma": float(gamma),
            "spot": float(spot),
            "T_grid": T_grid.tolist(),
            "theta_grid": theta_grid.tolist(),
        }

    def set_params(self, params: dict) -> None:
        """Restore model state from a snapshot produced by get_params()."""
        # core
        self.params = (
            float(params["rho"]),
            float(params["eta"]),
            float(params["gamma"]),
        )
        self._spot = float(params["spot"])

        # rebuild theta interpolator
        T_grid = np.asarray(params["T_grid"], float)
        theta_grid = np.asarray(params["theta_grid"], float)
        self.theta_interp = PchipInterpolator(T_grid, theta_grid, extrapolate=True)

    @staticmethod
    def build_params_dict(df_globals, df_knots):
        """
        Combine df_globals (rho, eta, gamma, spot) and df_knots (T, theta)
        into a nested dict keyed by date.

        Returns
        -------
        dict[Timestamp -> dict]
        """
        params_dict = {}
        for date, row in df_globals.iterrows():
            # Get knots for this date
            knots = df_knots.loc[date].sort_values("T")
            T_grid = knots["T"].to_list()
            theta_grid = knots["theta"].to_list()

            params_dict[pd.to_datetime(date)] = {
                "rho": float(row["rho"]),
                "eta": float(row["eta"]),
                "gamma": float(row["gamma"]),
                "spot": float(row.get("spot", np.nan)),  # if spot stored in df_globals
                "T_grid": T_grid,
                "theta_grid": theta_grid,
            }
        return params_dict

    def extract_surface_params(self, options):
        globals_list, knots_list = [], []

        for date, chain in options.groupby("date"):
            self.fit(chain)
            params = self.get_params()

            # 1) global SSVI params
            globals_list.append(
                {
                    "date": pd.to_datetime(date),
                    "rho": params["rho"],
                    "eta": params["eta"],
                    "gamma": params["gamma"],
                }
            )

            # 2) theta(T) knots
            for T, th in zip(params["T_grid"], params["theta_grid"]):
                knots_list.append({"date": pd.to_datetime(date), "T": T, "theta": th})

        df_globals = pd.DataFrame(globals_list).set_index("date")
        df_knots = pd.DataFrame(knots_list).set_index("date")

        return df_globals, df_knots
