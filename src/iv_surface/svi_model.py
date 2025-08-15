import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize
from .base_iv_surface_model import IVSurfaceModel


class SVI(IVSurfaceModel):
    """
    Per‐slice “raw” SVI with smooth interpolation of parameters across T.
    """
    param_names = ("a", "b", "rho", "m", "sigma")
    def __init__(self,
                 init_guess: dict = None,
                 weight_mode: bool = "atm",
                 butterfly_arb_strict: bool = False,
                 enforce_calendar_arb: bool = False,
                 calendar_arb_weight=50,
                 calendar_arb_kcut=0.02,
                 interpolation_mode=None,  # 'params' or 'w'
                 bounds: dict = None,
                 use_dynamic_init_guess: bool = True,
                 use_dynamic_bounds: bool = True,
                 glob_solver_fallback: bool = True,
                 rmse_iv_recalib: float = 0.02,
                 max_abs_iv_recalib: float = 0.05,
                 maturity_band: Tuple = (1/252, 2),
                 min_pts: int = 6,
                 params_interp_kind: str = 'linear'):
        super().__init__()

        # default SVI parameter bounds for raw‐SVI slice calibration
        default_bounds = {
            'a':     (1e-6,   1.0),      # total variance offset
            'b':     (1e-2,   1.0),     # variance slope
            'rho':   (-0.999, 0.999),  # correlation
            'm':     (-1.0,   1.0),   # log-moneyness shift 
            'sigma': (1e-3,   1.0),  # controls smile curvature
        }

        self.init_guess_user = init_guess or {} 
        self.weight_mode = weight_mode # decreasing weights from ATM to OTM
        self.butterfly_arb_strict = butterfly_arb_strict
        self.enforce_calendar_arb = enforce_calendar_arb 
        self.calendar_arb_weight = calendar_arb_weight # strength of penalty
        self.calendar_arb_kcut = calendar_arb_kcut
        self.interpolation_mode = interpolation_mode or (
            'w' if enforce_calendar_arb else 'params'
        )
        self.bounds = bounds or default_bounds
        self.use_dyn = use_dynamic_init_guess
        self.use_dynamic_bounds = use_dynamic_bounds
        self.glob_solver_fallback = glob_solver_fallback
        self.rmse_iv_recalib = rmse_iv_recalib
        self.max_abs_iv_recalib = max_abs_iv_recalib
        self.maturity_band = maturity_band
        self.params_interp_kind  = params_interp_kind
        self.min_pts  = min_pts  # min nb of calib points for the smile

        self.params_by_T  = {}      # raw SVI fits: T -> (a,b,rho,m,sigma)
        self._param_interp = {}    # after fit: name -> interp1d over T
        self.slice_stats = {}     # stores performance metrics per slice 


    # -------- Static helpers --------
    @staticmethod
    def _svi_raw(k, a, b, rho, m, sigma):
        """Raw SVI total‐variance formula:"""
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    

    # -------- Public API --------

    def fit(self, market_data: pd.DataFrame) -> None:
        """Calibrate one raw-SVI slice per quoted maturity and build
        interpolator between slices.
        """
        self._spot = market_data["underlying_last"].iloc[0]

        # ------- prepare dataframe ---------
        df  = self.prepare_iv_surface(market_data)
        df  = df[df["T"].between(*self.maturity_band)].copy()
        self._last_market = df                                  

         # ------- per-slice calibration---------
        T_vals, param_list = [], []
        prev_k = np.linspace(-0.4, 0.4, 61) # common k-grid for calendar test
        prev_params=None

        for T_i, sl in df.groupby("T"):
            m_lo, m_hi = self._moneyness_band(T_i)
            k_lo, k_hi = np.log(m_lo), np.log(m_hi)
            sl = sl[(sl["k"] >= k_lo) & (sl["k"] <= k_hi)]

            if len(sl) < self.min_pts:
                continue

            params = self._calibrate_raw_svi(
                sl["k"].values, sl["w"].values, T_i,
                prev_params=prev_params if self.enforce_calendar_arb else None,
                prev_k=prev_k
            )

            self.params_by_T[T_i] = params
            T_vals.append(T_i)
            param_list.append(params)
            prev_params = params  # update “previous” slice

        if not T_vals:
            raise RuntimeError("No maturities left after filters")

        # ------- unpack parameters----------
        T_grid = np.asarray(T_vals)              # already ascending
        A, B, R, M, S = map(np.array, zip(*param_list))

        # ----------  parameter interpolators -----------------
        self._param_interp = {
            "a"    : interp1d(T_grid, A, kind=self.params_interp_kind,
                            fill_value="extrapolate"),
            "b"    : interp1d(T_grid, B, kind=self.params_interp_kind,
                            fill_value="extrapolate"),
            "rho"  : interp1d(T_grid, R, kind=self.params_interp_kind,
                            fill_value="extrapolate"),
            "m"    : interp1d(T_grid, M, kind=self.params_interp_kind,
                            fill_value="extrapolate"),
            "sigma": interp1d(T_grid, S, kind=self.params_interp_kind,
                            fill_value="extrapolate"),
        }

    def pooled_rmse_iv(self):
        """Pooled RMSE in IV space across all quotes and maturities."""
        if not hasattr(self, "slice_stats") or not self.slice_stats:
            return float("nan")
        N  = np.array([v["n"] for v in self.slice_stats.values()], dtype=float)
        R2 = np.array([v["rmse_iv"]**2 for v in self.slice_stats.values()], dtype=float)
        return float(np.sqrt(np.sum(N * R2) / np.sum(N)))


    # -------- Plotting methods --------
    def plot_params(self, ax=None):
        params = pd.DataFrame.from_dict(
            self.params_by_T,
            orient='index',
            columns=self.param_names
        )
        params.index.name = 'T'
        params = params.sort_index().reset_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 7))

        for param in self.param_names:
            ax.plot(
                params['T'], params[param],
                marker='o', linestyle='-', label=param, alpha=0.8
            )

        ax.set_title('SVI Parameters vs. Maturity', fontsize=14)
        ax.set_xlabel('T (years)', fontsize=12)
        ax.set_ylabel('Parameter value', fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        return ax
    
    def plot_total_variance_slices(self,
                                n_slices: int = 6,
                                k_band: tuple = (-0.2, 0.15),
                                nk: int = 201,
                                show_market: bool = True,
                                ax=None):
        if not self.params_by_T:
            raise RuntimeError("Fit the model first (params_by_T is empty).")

        # maturities sorted (smallest first) and take first n
        Ts = np.array(sorted(self.params_by_T.keys()))
        Ts = Ts[::5][:n_slices]
        kmin, kmax = k_band
        kg = np.linspace(kmin, kmax, nk)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.axvline(0.0, color='gray', lw=1.0, alpha=0.6)

        for T_i in Ts:
            a,b,rho,m,sigma = self.params_by_T[T_i]
            w_model = self._svi_raw(kg, a,b,rho,m,sigma)
            ax.plot(kg, w_model, lw=2.0, label=fr"T={T_i:.4f} (~{int(round(T_i*252))}d)")

            if show_market and hasattr(self, "_last_market") and self._last_market is not None:
                sl = self._last_market[self._last_market["T"] == T_i]
                if len(sl):
                    mask = (sl["k"] >= kmin) & (sl["k"] <= kmax)
                    sl = sl.loc[mask]
                    ax.scatter(sl["k"], sl["w"], s=10, alpha=0.6)

        ax.set_xlabel(r"Log moneyness $k$")
        ax.set_ylabel("Implied Total Variance")
        ax.legend(title="", fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return ax
    
    def plot_slice_error_metrics(self, ax=None, x_unit="days", as_percent=True,
                                ref_lines=(0.5, 1.0)):
        
        if not self.slice_stats:
            raise RuntimeError("Fit the model first (slice_stats is empty).")

        # Collect and sort by maturity
        Ts = np.array(sorted(self.slice_stats.keys()), dtype=float)  # years
        rmse = np.array([self.slice_stats[T].get("rmse_iv", np.nan) for T in Ts], dtype=float)
        maxabs = np.array([self.slice_stats[T].get("max_abs_iv", np.nan) for T in Ts], dtype=float)

        if x_unit == "days":
            X = Ts * 252.0
            xlab = "DTE (days)"
        elif x_unit == "years":
            X = Ts
            xlab = "T (years)"
        else:
            raise ValueError("x_unit must be 'days' or 'years'")

        # Y as percent if requested
        yscale = 100.0 if as_percent else 1.0
        ylab = "IV error (%)" if as_percent else "IV error (abs.)"

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        else:
            if not isinstance(ax, (list, tuple)) or len(ax) != 2:
                raise ValueError("ax must be a tuple/list of two Axes (top, bottom).")
            ax1, ax2 = ax

        # Top: RMSE
        ax1.plot(X, rmse * yscale, marker="o", lw=1.8, label="RMSE (IV)")
        if ref_lines:
            for r in ref_lines:
                ax1.axhline(r if as_percent else r/100.0, ls="--", lw=0.8, color="gray")
        ax1.set_ylabel(ylab)
        ax1.set_title("Per-slice IV fit errors vs maturity")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Bottom: Max |error|
        ax2.plot(X, maxabs * yscale, marker="o", lw=1.8, color="C1", label="Max |IV error|")
        if ref_lines:
            for r in ref_lines:
                ax2.axhline(r if as_percent else r/100.0, ls="--", lw=0.8, color="gray")
        ax2.set_xlabel(xlab)
        ax2.set_ylabel(ylab)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        return ax1, ax2


    # -------- Private helpers --------

    def _default_vector(self, k, w):
        """Return the 5-vector of slice-specific defaults."""
        if self.use_dyn:
            a0 = max(0.0, w.min())
            b0 = min((w.max()-w.min())/ (np.ptp(k)+1e-6), 0.10)
            m0 = k[np.argmin(w)]
            sigma0 = min(np.std(k)*0.1 + 1e-3, 0.10)
            dw_dk = np.gradient(w, k)[np.argmin(np.abs(k))]
            rho0  = np.clip(dw_dk/b0, -0.8, 0.4) if b0>0 else 0.0
        else:                              # static fall-back numbers
            a0, b0, rho0, m0, sigma0 = 0.01, 0.10, 0.0, 0.0, 0.05
        return dict(a=a0, b=b0, rho=rho0, m=m0, sigma=sigma0)

    def _initial_guesses(self, k, w):
        """Yield one np.array per Cartesian combination."""
        base = self._default_vector(k, w)

        # merge user overrides (scalar or iterable)
        param_lists = {}
        for name in self.param_names:
            if name in self.init_guess_user:
                user_val = self.init_guess_user[name]
                param_lists[name] = (user_val if isinstance(user_val,
                                         (list,tuple,np.ndarray))
                                     else [user_val])
            else:
                param_lists[name] = [base[name]]        # single default

        # Cartesian product over only those lists with >1 element
        for combo in itertools.product(*(param_lists[p] for p in self.param_names)):
            yield np.array(combo, float)

    def _slice_objective(self, params: np.ndarray,
                         k: np.ndarray,
                         w: np.ndarray,
                         T: float,
                         prev_params=None,
                         prev_k=None) -> float:
        """
        Objective for one raw‐SVI slice: weighted SSE + optional
        calendar & butterfly arbitrage constraints .
        """
        a, b, rho, m, sigma = params
        model = self._svi_raw(k, a, b, rho, m, sigma)
        resid = model - w

        # -------------  weighted SSE -----------------
        weights = self._weights(k, T)
        sse = float(np.sum(weights * resid**2))

        # ------------- strict butterfly (Ferhani 2020) ----
        if self.butterfly_arb_strict:
            # wing slopes
            C_p, C_m  = b * (rho + 1), b * (rho - 1)
            # intercepts
            D_p, D_m  = a - m * C_p, a - m * C_m
            
            # butterfly check
            ok = (0 < C_p**2 < 4 and 0 < C_m**2 < 4 and
                D_p*(4-D_p) > C_p**2 and D_m*(4-D_m) > C_m**2)
            if not ok:
                return 1e10
            
        # ------------- enforce calendar arbitrage between slices -------
        if self.enforce_calendar_arb and prev_params is not None:
            # evaluate both slices on the shared k-grid
            a0,b0,r0,m0,s0 = prev_params
            w_prev = self._svi_raw(prev_k, a0,b0,r0,m0,s0)
            w_curr = self._svi_raw(prev_k, a ,b ,rho,m ,sigma)

            # hinge: only penalize where current < previous
            viol = np.maximum(w_prev - w_curr, 0.0)

            # focus on the right wing if you want
            if self.calendar_arb_kcut is not None:
                wing_mask = (prev_k >= self.calendar_arb_kcut).astype(float)
                viol *= wing_mask

            # scale-invariant weight
            lam = self.calendar_arb_weight / max(np.mean(w_prev), 1e-8)
            sse += lam * float(np.sum(viol**2))

        return sse
    
    def _weights(self, k, T):
        if self.weight_mode == "uniform":
            return np.ones_like(k)
        if self.weight_mode == "atm" and T <= 0.05:
            return 1.0/(np.abs(k)+0.01)**2
        if self.weight_mode == "atm":
            return 1.0/(np.abs(k)+0.05)
        raise ValueError("weight_mode must be 'uniform'|'atm'")
    
    def _calibrate_raw_svi(self, k, w, T, prev_params=None, prev_k=None):
        lb = np.array([self.bounds[n][0] for n in self.param_names])
        ub = np.array([self.bounds[n][1] for n in self.param_names])

        obj = lambda p: self._slice_objective(
            p, k, w, T,
            prev_params=prev_params,
            prev_k=prev_k
        )

        if self.use_dynamic_bounds:
            ub[0] = 0.5*np.max(w)  # a <= 1/2*max(w)
            lb[3] = 2*np.min(k)    # 2*np.min(k) <= m <= 2*np.max(k)
            ub[3] = 2*np.max(k)
            lb[1] = 1e-2

            if T < 0.05: # ≈ 13 DTE
                lb[3], ub[3] = -0.3, 0.3  # tighter m-bounds 

        # ------------- local solver -------------
        best = None
        for x0 in self._initial_guesses(k, w):
            x0 = np.clip(x0, lb, ub)
            sol = minimize(fun=obj,
                           x0=x0, 
                           bounds=list(zip(lb,ub)),
                           method="L-BFGS-B",
                           options={"maxiter": 1000})
            if sol.success and (best is None or sol.fun < best.fun):
                best = sol

        if best is None:
            raise RuntimeError("SVI slice failed for all initial guesses")
    
        # metrics for local solution
        local_params  = best.x
        local_metrics = self._compute_slice_metrics(k, w, T, local_params)

        # early exit if good enough or no global fallback
        if (not self.glob_solver_fallback) or (
            local_metrics["rmse_iv"] < self.rmse_iv_recalib and
            local_metrics["max_abs_iv"] < self.max_abs_iv_recalib
        ):
            self.slice_stats[T] = local_metrics
            return tuple(local_params)           

        # ------------- global DE + polish  -----------
        de = differential_evolution(
            obj,
            bounds=list(zip(lb,ub)),
            strategy='best1bin', popsize=15, maxiter=150,
            tol=1e-6,  mutation =(0.5, 1.0), recombination=0.7, 
            polish=False, # we'll polish ourselves
            disp=False,
            updating ='deferred' # faster in recent SciPy versions
        )
        pol = minimize(
            fun=obj,
            x0=de.x, 
            bounds=list(zip(lb,ub)),
            method='L-BFGS-B', options={'maxiter':500}
        )

        # choose final by objective value (and compute metrics for it)
        if pol.success and pol.fun < best.fun:
            final_params  = pol.x
            final_metrics = self._compute_slice_metrics(k, w, T, final_params)
        else:
            final_params  = local_params
            final_metrics = local_metrics

        self.slice_stats[T] = final_metrics
        return tuple(final_params)

    def _w_between_slices(self, k: np.ndarray, T: np.ndarray, atol=1e-10):
        """
        Calendar-safe interpolation of total variance in T between raw-SVI slices.
        """
        k = np.atleast_1d(k).astype(float)
        Tq = np.atleast_1d(T).astype(float)

        # sorted slices and params
        Ts = np.array(sorted(self.params_by_T.keys()))
        if Ts.size < 2:
            raise RuntimeError("Need at least two calibrated slices to interpolate in T.")
        
        # for each query T, find bracketing indices
        idx = np.searchsorted(Ts, Tq, side="right") - 1
        idx = np.clip(idx, 0, len(Ts)-2)
        T0 = Ts[idx]
        T1 = Ts[idx+1]

        # snap-to-knot: if T is (numerically) exactly a slice, evaluate that slice
        exact0 = np.isclose(Tq, T0, atol=atol)
        exact1 = np.isclose(Tq, T1, atol=atol)

        lam = (Tq - T0) / (T1 - T0 + 1e-16)
        lam = np.maximum(lam, 0.0) # clamp to 0 below T0 and linear to T above T1

        # gather parameters for both slices
        P0 = np.array([self.params_by_T[t] for t in T0]) # (nT, 5)
        P1 = np.array([self.params_by_T[t] for t in T1])

        # evaluate both slices on the same k-mesh
        K, _ = np.meshgrid(k, Tq, indexing="xy") # (nT, nK)
        A0,B0,R0,M0,S0 = (P0[:,0,None], P0[:,1,None], P0[:,2,None], P0[:,3,None], P0[:,4,None])
        A1,B1,R1,M1,S1 = (P1[:,0,None], P1[:,1,None], P1[:,2,None], P1[:,3,None], P1[:,4,None])
        W0 = self._svi_raw(K, A0, B0, R0, M0, S0)
        W1 = self._svi_raw(K, A1, B1, R1, M1, S1)
        W  = (1.0 - lam)[:,None]*W0 + lam[:,None]*W1 # (nT, nK)

        # Overwrite exact rows with the exact slice evaluation (no FP drift)
        if exact0.any():
            rows = np.where(exact0)[0]
            for r in rows:
                W[r,:] = self._svi_raw(k, *self.params_by_T[T0[r]])
        if exact1.any():
            rows = np.where(exact1)[0]
            for r in rows:
                W[r,:] = self._svi_raw(k, *self.params_by_T[T1[r]])

        return W

    def _implied_total_variance(self, k, T):
        k = np.atleast_1d(k).astype(float)
        T = np.atleast_1d(T).astype(float)

        if self.interpolation_mode == 'params': # could use PCHIP instead
            # classic: interpolate a,b,rho,m,sigma then evaluate raw SVI
            a = self._param_interp['a'](T); b = self._param_interp['b'](T)
            rho = self._param_interp['rho'](T); m = self._param_interp['m'](T)
            sig = self._param_interp['sigma'](T)
            K,_ = np.meshgrid(k, T, indexing='xy')
            return self._svi_raw(K, a[:,None], b[:,None], rho[:,None], m[:,None], sig[:,None])

        elif self.interpolation_mode == 'w':
            # calendar-safe: piecewise-linear blend of w(k,T) between slices
            return self._w_between_slices(k, T)

        else:
            raise ValueError("interpolation_mode must be 'params' or 'w'")
        
    def _compute_slice_metrics(self, k, w, T, params):
        """Compute IV / w errors for a slice and return a dict."""
        a,b,rho,m,sigma = params
        w_model  = self._svi_raw(k, a,b,rho,m,sigma)

        # guard small T
        T_eff    = max(float(T), 1e-16)
        sig_mkt  = np.sqrt(np.maximum(w, 0.0) / T_eff)
        sig_mdl  = np.sqrt(np.maximum(w_model, 0.0) / T_eff)

        err_iv   = sig_mkt - sig_mdl
        err_w    = w - w_model

        metrics = dict(
            n        = int(len(k)),
            rmse_iv  = float(np.sqrt(np.mean(err_iv**2))),
            max_abs_iv= float(np.max(np.abs(err_iv))),
            rmse_w   = float(np.sqrt(np.mean(err_w**2))),
        )
        return metrics