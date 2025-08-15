import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from matplotlib.colors import Normalize


class IVSurfaceModel(ABC):
    """Abstract base for any IV-surface model."""
    def __init__(self):
        self._spot: float = None
        self._last_market = None
        
    @staticmethod
    def prepare_iv_surface(options: pd.DataFrame):
        df = options.copy()

        # choose OTM vol
        df['iv_smile'] = np.where(
            df['strike'] < df['underlying_last'], 
            df['p_iv'],
            np.where(
                df['strike'] > df['underlying_last'], 
                df['c_iv'], 
                0.5*(df['c_iv'] + df['p_iv'])
            )
        )

        # log-forward moneyness
        df['k'] = np.log(df['strike'] / df['underlying_last'])

        # total implied variance
        df['w'] = df['iv_smile']**2 * df['T']

        return df
    
    @staticmethod
    def _moneyness_band(T):
        """Return maturity bands (m_lo, m_hi) given T in years."""
        days = T * 252
        if days < 7:
            return 0.90, 1.05
        elif days < 30:
            return 0.85, 1.15
        elif days < 180:
            return 0.80, 1.20
        else:
            return 0.70, 1.30

    @abstractmethod
    def fit(self, market_data: pd.DataFrame) -> None:
        """Calibrate to raw market quotes (DataFrame of k, T, σ_mkt)."""
        pass

    @abstractmethod
    def _implied_total_variance(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return total variance w(k,T)."""
        pass
    
    def implied_vol(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return implied volatility as sqrt(w(k,T) / T)"""
        if self._last_market is None:
            raise RuntimeError("You must .fit(...) first to")

        K_arr = np.atleast_1d(K).astype(float) # (nK,)
        T_arr = np.atleast_1d(T).astype(float) # (nT,)
        spot = self._last_market["underlying_last"].iloc[0]
        k_arr = np.log(K_arr / spot)

        W = self._implied_total_variance(k_arr, T_arr) # (nT, nK)

        if W.ndim == 2 and T_arr.ndim == 1:
            T_arr = T_arr[:, None] # (nT, 1)

        iv = np.sqrt(W / T_arr)
        return iv.item() if iv.size == 1 else iv
    
    def check_arbitrage(self,
                        n_k:int=201,           
                        n_T:int=120,          
                        k_pad:float=0.05,     
                        tol:float=1e-10    
                    ):
        """Static-arbitrage diagnostics."""
        if self._last_market is None:
            raise RuntimeError("You must .fit(...) first to")
        
        # ---------- build k,T grid covering fitted region -----------
        df = self._last_market
        k = df["k"]

        T_slice = np.array(sorted(df["T"].unique()))
        T_min, T_max = T_slice.min(), T_slice.max()
        T_grid = np.concatenate([T_slice,
                                np.linspace(T_min, T_max, n_T)])
        T_grid = np.unique(np.round(T_grid, 6)) 

        k_min, k_max = k.min() - k_pad, k.max() + k_pad
        k_grid = np.linspace(k_min, k_max, n_k)

        W = self._implied_total_variance(k_grid, T_grid)

        # ---------- butterfly test: ∂²w/∂k² >= 0 --------------
        dk = k_grid[1]-k_grid[0]
        dw2_dk2 = (W[:,2:] - 2*W[:,1:-1] + W[:,:-2]) / dk**2
        butterfly_ok = (dw2_dk2 >= -tol)

        # ---------- calendar test: ∂w/∂T >= 0 -----------------
        dT = np.diff(T_grid)
        dw_dT = (np.diff(W, axis=0).T / dT).T
        calendar_ok = (dw_dT >= -tol)

        # ---------- summary report -----------------------------
        bad_bfly = np.where(~butterfly_ok)
        bad_cal  = np.where(~calendar_ok)
        rows = []
        for r,c in zip(*bad_bfly):
            rows.append({"type":"butterfly",
                        "T":T_grid[r],
                        "k":k_grid[c+1],
                        "value":dw2_dk2[r,c]})
        for r,c in zip(*bad_cal):
            rows.append({"type":"calendar",
                        "T":(T_grid[r]+T_grid[r+1])/2,
                        "k":k_grid[c],
                        "value":dw_dT[r,c]})

        report = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=["type","T","k","value"])

        return report
    
    def plot_smiles(
            self,
            target_days=[1,5,10,15,20,25,30,40,50,60,126,252],
            total_var=False,
    ):
        """3×4 grid of IV(k) using the maturities closest to `target_days`."""
        if self._last_market is None:
            raise RuntimeError("You must .fit(...) first to")
        df = self._last_market.copy()
        df["days"] = (df["T"] * 252).round().astype(int)

        def closest_distinct(avail_days, targets):
            avail = np.asarray(sorted(avail_days))
            chosen = []
            for t in targets:
                if avail.size == 0:
                    chosen.append(None)          
                    continue
                idx = np.abs(avail - t).argmin()
                chosen.append(int(avail[idx]))
                avail = np.delete(avail, idx)  
            return chosen

        # choose the nearest available maturity for each target
        chosen_days = closest_distinct(df["days"].unique(), target_days)
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        axes = axes.flatten()

        for ax, D in zip(axes, chosen_days):
            if D is None:          
                ax.axis("off")
                continue

            T_i = D / 252.0
            slice_df  = df[np.isclose(df["T"], T_i)]
            
            m_lo, m_hi = self._moneyness_band(T_i)
            k_lo, k_hi = np.log(m_lo), np.log(m_hi)
            slice_df = slice_df[
                (slice_df["k"] >= k_lo) & 
                (slice_df["k"] <= k_hi)
            ].copy()
            
            k = slice_df["k"].values

            if total_var:
                model = slice_df["w"].values
                market = self._implied_total_variance(k, T_i).ravel()
            else:
                model = slice_df["iv_smile"].values
                market = self.implied_vol(slice_df["strike"], T_i).ravel()
            label = "TIVar" if total_var else "IV"

            ax.scatter(k, model, s=18, alpha=0.8, color="blue", label=f"Market {label}")
            ax.plot(k, market, color="red", label=f"Model {label}")
            ax.axvline(0, lw=.6, c="black", linestyle="--")

            ax.set_title(f"DTE = {int(D)}", fontsize=12)
            ax.set_xlabel("Log-Forward Moneyness (k)")
            title = "Total Implied Variance" if total_var else "Implied Volatility (IV)"
            ax.set_ylabel(f"{title}")
            ax.legend(loc="upper right")

        title = "SVI vs Market "
        title += "Total Implied Variances" if total_var else "Implied Volatilities"
        fig.suptitle(title, fontsize=18, y=1.02)
        fig.tight_layout()
        plt.show()

    def plot_iv_surface(
        self,
        method_name=None,
        moneyness_band=(0.8, 1.2),
        dte_band=(1, 60),
        ax=None,
        n_k=50,
        n_T=50,
        n_term_lines=6,  
        n_skew_lines=6,
        cmap_name="turbo",
        colorbar=True,
        elev=30,
        azim=310,
    ):
        if self._last_market is None:
            raise RuntimeError("Call fit(...) first.")

        df   = self._last_market.copy()
        spot = float(df["underlying_last"].iloc[0])

        # Build grids
        d_min, d_max = dte_band
        T_grid = np.linspace(d_min/252.0, d_max/252.0, n_T)       # years
        k_lo, k_hi = np.log(moneyness_band[0]), np.log(moneyness_band[1])
        k_grid = np.linspace(k_lo, k_hi, n_k)
        K_grid = spot * np.exp(k_grid)

        # Evaluate model IV (shape: n_T x n_k)
        IV = self.implied_vol(K_grid, T_grid) * 100.0  # percent
        Kmesh, Tmesh = np.meshgrid(k_grid, T_grid, indexing="xy")

        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        
        # Dark theme
        fig.patch.set_facecolor("#0e0f12")
        ax.set_facecolor("#0e0f12")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((0.06, 0.07, 0.08, 0.0))
            pane.set_edgecolor((1, 1, 1, 0.25))
        ax.tick_params(colors="#d8d8d8")

        # Surface
        cmap = plt.get_cmap(cmap_name)
        surf = ax.plot_surface(
            Kmesh, Tmesh*252.0, IV,
            cmap=cmap, linewidth=0.1, antialiased=True, alpha=0.95, rstride=1, cstride=1
        )

        ax.view_init(elev=elev, azim=azim)

        # ----- add padding around the surface -----
        x_min, x_max = k_grid.min(), k_grid.max()
        pad_x = 0.4
        dx = x_max - x_min

        # Grid styling
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            g = axis._axinfo["grid"]
            g["color"] = (1, 1, 1, 0.15)
            g["linewidth"] = 0.4

        # ----- Left wall: term-structure curves at fixed k -----
        k_lines = np.quantile(k_grid, np.linspace(0.05, 0.95, n_term_lines))
        norm_k  = Normalize(vmin=k_grid.min(), vmax=k_grid.max())
        y_ts    = T_grid * 252.0
        x_wall  = k_grid.min()  
        x_wall  = (x_min - 0.5*pad_x*dx)

        for k0 in k_lines:
            K0 = spot * np.exp(k0)
            iv_line = (self.implied_vol(np.array([K0]), T_grid).ravel()) * 100.0
            ax.plot(
                y_ts, iv_line, zs=x_wall, zdir="x",
                color=cmap(norm_k(k0)), lw=2.0, alpha=0.95
            )

        # ----- Back wall: skew curves at fixed T -----
        T_lines = np.quantile(T_grid, np.linspace(0.05, 0.95, n_skew_lines))
        norm_T  = Normalize(vmin=T_grid.min(), vmax=T_grid.max())
        y_back = (T_grid.max() + 0.2*(T_grid.max() - T_grid.min())) * 252.0

        for T0 in T_lines:
            iv_line = (self.implied_vol(K_grid, float(T0)).ravel()) * 100.0
            ax.plot(
                k_grid, iv_line,
                zs=y_back, zdir="y",             
                color=cmap(norm_T(T0)), lw=2.0, alpha=0.95
            )

        ax.set_xlabel("Log-moneyness  k", color="#eaeaea")
        ax.set_ylabel("Expiration (days)", color="#eaeaea")
        ax.set_zlabel("Implied Vol (%)", color="#eaeaea")
        title = 'Implied Volatility Surface'
        title += f": {method_name}" if method_name else ""
        if method_name:
            ax.set_title(title, fontsize=16, y=1.03, color="white")

        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(IV)
        if colorbar:
            cbar = plt.colorbar(m, ax=ax, pad=0.1, shrink=0.5, aspect=10)
            cbar.ax.tick_params(colors="#eaeaea")
            cbar.set_label("IV (%)", color="#eaeaea")
            
        return ax