import numpy as np
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
            (-0.999,0.999),   # ρ
            (1e-3, 2.0),     # η
            (1e-3, 0.999)   # γ
        ]
        self.bounds = bounds or default_bounds

    @classmethod
    def _w(cls, k, theta, params):
        rho, eta, gamma = params
        phi = cls._phi(theta, eta, gamma)
        return 0.5*theta*(1 + rho*phi*k + np.sqrt((phi*k + rho)**2 + 1 - rho**2))

    def _objective(self, k, theta, w, weights, params):
        rho, eta, gamma = params
        w_model = self._w(k, theta, params)
        sse = np.sum(weights * (w_model - w)** 2)

        # ---- soft no-arb penalties (butterfly + calendar) ----
        # Butterfly (wing) caps (sufficient):
        theta_min, theta_max = np.min(theta), np.max(theta)

         # θφ(θ)(1+|ρ|) < 4
        eta_cap_3 = 4.0 / ((1.0 + abs(rho)) * theta_max**(1.0 - gamma))

        # θφ(θ)^2(1+|ρ|) ≤ 4
        eta_cap_4 = 2.0 / np.sqrt((1.0 + abs(rho)) * theta_min**(1.0 - 2.0*gamma))

        eta_cap = min(eta_cap_3, eta_cap_4)
        if self.butterfly_arb_strict and eta >= eta_cap:
            sse += 1e8 * (eta - eta_cap)**2   # hinge penalty

        return sse
    