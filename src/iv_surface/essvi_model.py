import numpy as np
import matplotlib.pyplot as plt
from .base_xssvi import xSSVI


class eSSVI(xSSVI):
    """
    eSSVI (extended SSVI) fit:
    w(k,T) = θ(T)/2 (1 + ρ(θ) φ(θ) k + sqrt((φ(θ) k + ρ(θ))^2 + 1 - ρ(θ)²))
    
    φ(θ) = η θ^{-γ}
    ρ(θ) = tanh(a + bθ)
    """
    def __init__(self, bounds=None, **kwargs):
        super().__init__(**kwargs)
        default_bounds = [
            (-3.0, 3.0),      # alpha (for _rho)
            (-10.0, 10.0),   # beta (for _rho)
            (1e-4, 2.0),    # eta (for _phi)
            (1e-3, 0.999)  # gamma (for _phi)
        ] 

        self.bounds = bounds or default_bounds

    # -------- kernel pieces --------
    @staticmethod
    def _phi(theta, eta, gamma):
        return eta * np.power(theta, -gamma)
    
    @staticmethod
    def _rho(theta, alpha, beta):
        # bounded in (-1,1), smooth, monotone if beta has a sign
        return np.tanh(alpha + beta * theta)

    @classmethod
    def _w(cls, k, theta, params):
        alpha, beta, eta, gamma = params
        rho = cls._rho(theta, alpha, beta)
        phi = cls._phi(theta, eta, gamma)
        return 0.5*theta * (1.0 + rho*phi*k + np.sqrt((phi*k + rho)**2 + 1.0 - rho**2))

    def _objective(self, k, theta, w, weights, params):
        alpha, beta, eta, gamma = params
        w_model = self._w(k, theta, params)
        sse = np.sum(weights * (w_model - w)** 2)

        # ---- soft butterfly arb penalties ----
        # cap by max rho and min/max theta:
        rho_all = self._rho(theta, alpha, beta)
        rho_max  = float(np.max(np.abs(rho_all)))
        theta_min, theta_max = np.min(theta), np.max(theta)
        eta_cap_3 = 4.0 / ((1.0 + rho_max) * theta_max**(1.0 - gamma))
        eta_cap_4 = 2.0 / np.sqrt((1.0 + rho_max) * theta_min**(1.0 - 2.0*gamma))
        eta_cap = min(eta_cap_3, eta_cap_4)

        if self.butterfly_arb_strict and eta >= eta_cap:
            sse += 1e8 * (eta - eta_cap)**2   # hinge penalty
            
        return sse
    
    def plot_rho(self, n: int = 200, ax=None):
        if self._last_market is None:
            raise RuntimeError("Call fit(...) first.")
        T_min = float(self._last_market["T"].min())
        T_max = float(self._last_market["T"].max())
        T_grid = np.linspace(T_min, T_max, n)
        theta = self.theta_interp(T_grid)
        alpha, beta, _, _ = self.params
        rho_vals = self._rho(theta, alpha, beta)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(theta, rho_vals, lw=2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\rho(\theta)$")
        ax.set_title(r"eSSVI rho curve $\rho(\theta)$")
        return ax