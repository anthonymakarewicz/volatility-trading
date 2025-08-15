import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod
from typing import Tuple
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import PchipInterpolator
from .base_iv_surface_model import IVSurfaceModel

class xSSVI(IVSurfaceModel):
    """Intermediary abstract class for SSVI and eSSVI.
    Need to define the virtual method _objective and _w"""
    def __init__(self,
                 weight_mode: str = "atm",
                 butterfly_arb_strict: bool = False,
                 maturity_band : Tuple = (1/252, 2),
                 moneyness_band: Tuple = (0.75, 1.25),
                 min_pts: int = 6
        ):
        super().__init__()
        self.butterfly_arb_strict = butterfly_arb_strict
        self.weight_mode = weight_mode
        self.maturity_band  = maturity_band
        self.moneyness_band = moneyness_band
        self.min_pts = min_pts

        self.theta_interp = None
        self.params = None # Store parameters of SSVi or eSSVI
        self.bounds = None

    @staticmethod
    @abstractmethod
    def _w(k, theta, params):
        pass
    
    @abstractmethod
    def _objective(self, k, theta, w, weights, params):
        pass

    @staticmethod
    def _phi(theta, eta, gamma):
        return eta * theta**(-gamma)

    def fit(self, market_data: pd.DataFrame) -> None:
        self._spot = market_data["underlying_last"].iloc[0]
        df = self.prepare_iv_surface(market_data)

        # ------- log moneyness and maturity filters -------
        tmin, tmax = self.maturity_band
        df = df[df["T"].between(tmin, tmax)]
        mlo, mhi = self.moneyness_band
        klo, khi = np.log(mlo), np.log(mhi)
        df = df[df["k"].between(klo, khi)]
        if self.min_pts > 0:
            df = df.groupby("T").filter(lambda g: len(g) >= self.min_pts)

        if df.empty:
            raise RuntimeError("No quotes after filters")
        self._last_market = df.copy()

        # ------ θ̂(T): ATM quadratic intercept per slice, then PCHIP & make monotone -----
        groups = {T: g for T, g in df.groupby("T")}
        T_grid = np.array(sorted(groups.keys()))
        theta_hat = []
        for T in T_grid:
            g = groups[T]
            th = self._estimate_theta_slice(g["k"].to_numpy(), g["w"].to_numpy())
            theta_hat.append(th)    
        theta_hat = np.asarray(theta_hat, float)
        theta_hat  = np.maximum.accumulate(theta_hat)                 # monotone
        self.theta_interp = PchipInterpolator(T_grid, theta_hat, extrapolate=True)

        # ------- stack quotes ------
        k_all = df["k"].to_numpy()
        T_all = df["T"].to_numpy()
        w_all = df["w"].to_numpy()
        theta_all = self.theta_interp(T_all)
        weights = self._weights(k_all, theta_all)

        # cache θ range for constraints
        self._theta_min = float(theta_all.min())
        self._theta_max = float(theta_all.max())

        # ------- global solver + polish ------
        obj = lambda p: self._objective(k_all, theta_all, w_all, weights, np.asarray(p))

        de = differential_evolution(
            obj, bounds=self.bounds,
            strategy="best1bin", popsize=20, maxiter=200,
            tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            polish=False, disp=False
        )
        if not de.success:
            raise RuntimeError(de.message)
        
        pol = minimize(
            obj, de.x, method="L-BFGS-B",
            bounds=self.bounds, options={"maxiter": 500}
        )
        x = pol.x if pol.success and pol.fun <= de.fun else de.x

        self.params = tuple(x)

    def _weights(self, k, theta):
        if self.weight_mode == "uniform":
            return np.ones_like(k)
        if self.weight_mode == "atm":
            return 1.0/(np.abs(k)+0.05)
        if self.weight_mode == "atm_theta":
            return 1.0/(np.abs(k)+0.05) / np.sqrt(theta + 1e-8)
        raise ValueError("weight_mode must be 'uniform'|'atm'|'atm_theta'")

    def _estimate_theta_slice(self, k, w, band=0.05):
        """Fits a local quadratic around the ATM and extratc the intercept
        for θ̂(T), else fallback to the minimum w for θ̂(T)"""
        m = np.abs(k) <= band
        if m.sum() < 5:
            return np.min(w)  # fallback
        X = np.c_[np.ones(m.sum()), k[m], k[m]**2]
        wt = 1.0/(np.abs(k[m])+1e-3)               # ATM-heavy weights
        beta = np.linalg.lstsq(X*wt[:,None], (w[m]*wt), rcond=None)[0]
        return float(beta[0])                       # intercept ~= θ̂
    
    def _implied_total_variance(self,
                                k: np.ndarray,
                                T: np.ndarray
                            ) -> np.ndarray:
        """Return w(k,T) with shape (nT, nK)."""
        k_arr = np.atleast_1d(k).astype(float)
        T_arr = np.atleast_1d(T).astype(float)

        # θ(T) from the monotone PCHIP
        theta = self.theta_interp(T_arr)  # (nT,)

        # broadcast via meshgrid
        K, Theta = np.meshgrid(k_arr, theta, indexing="xy") # (nT, nK)

        return self._w(K, Theta, self.params) # (nT, nK)
    
    def pooled_rmse_iv(self) -> float: # Could put it in IVSurfaceModel
        if self._last_market is None:
            raise RuntimeError("Call fit(...) first; _last_market is empty.")

        df = self._last_market
        sse = 0.0
        N = 0.0

        for T_i, sl in df.groupby("T"):
            iv_market = sl["iv_smile"]
            iv_model = self.implied_vol(sl["strike"].to_numpy(), float(T_i)).ravel()
            err = iv_market - iv_model
            sse += float(np.sum(err**2))
            N += len(sl)

        return float(np.sqrt(sse / max(N, 1e-16)))
    
    def plot_theta(self,
               n: int = 300,
               ax=None,
               as_dte: bool = True
        ):
        """Plot the fitted ATM total variance curve θ(T)."""
        if self.theta_interp is None:
            raise RuntimeError("Call fit(...) first; theta_interp is None.")
        if self._last_market is None:
            raise RuntimeError("Call fit(...) first; _last_market is empty.")

        T_min = self._last_market["T"].min()
        T_max = self._last_market["T"].max()
        if not (T_max > T_min):
            raise ValueError("T_max must be greater than T_min.")

        # Build grid and evaluate θ(T)
        T_grid = np.linspace(T_min, T_max, n)
        theta_vals = self.theta_interp(T_grid)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))

        x_vals = T_grid * 252.0 if as_dte else T_grid

        # Plot θ(T) curve
        ax.plot(x_vals, theta_vals, lw=2)

        # Plot observed θ̂(T) ATM Total variance
        T_obs = np.array(sorted(self._last_market["T"].unique()))
        theta_obs = self.theta_interp(T_obs)
        x_obs = T_obs * 252.0 if as_dte else T_obs
        ax.scatter(x_obs, theta_obs, s=25, alpha=1, color="orange", label=r"Fitted $\theta(T)$")
        ax.legend(loc="upper left")
        ax.set_xlabel("DTE (days)" if as_dte else "Maturity T (years)")
        ax.set_ylabel(r"$\theta(T)$")
        ax.set_title(fr"{self.__class__.__name__} ATM total variance term-structure $\theta(T)$")

        return ax
    
    def plot_phi(self, n: int = 200, ax=None):
        """Plot the fitted vol-vol curve φ(θ)"""
        if self._last_market is None:
            raise RuntimeError("Call fit(...) first.")
        T_min = float(self._last_market["T"].min())
        T_max = float(self._last_market["T"].max())
        T_grid = np.linspace(T_min, T_max, n)
        theta = self.theta_interp(T_grid)
        _, eta, gamma = self.params
        phi_vals = self._phi(theta, eta, gamma)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))

        ax.plot(theta, phi_vals, lw=2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\phi(\theta)$")
        ax.set_title(fr"{self.__class__.__name__} vol-vol curve $\phi(\theta)$")
        if np.all(phi_vals > 0):
            ax.set_yscale("log")
            ax.set_ylabel(r"$\log(\phi(\theta))$")
        return ax