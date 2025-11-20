import numpy as np
import pandas as pd
import yfinance as yf
from .vol_estimators import rv_intraday
from ..options.greeks import solve_strike_for_delta
from .macro_features import create_macro_features
from typing import Optional, Tuple

def create_forward_target(
    daily_variance: pd.Series, 
    horizon: int = 21
) -> pd.Series:
    """
    daily_variance: per-day realized variance (NOT sqrt, NOT annualized).
                    index = trading days.
    horizon: forecast horizon in trading days (e.g. 21 ~ 1 month).

    Returns:
        y_t = log( average future variance from t+1 to t+horizon )
    """
    # trailing mean, then realign to t as forward mean
    fwd_avg_var = (
        daily_variance
        .rolling(horizon).mean()       # at t+h: mean of [t+1 .. t+h]
        .shift(-horizon)               # move it back so it's aligned at t
    )

    y = np.log(fwd_avg_var)
    y.name = f"log_fwd_var_{horizon}d"
    return y


def create_har_lags(real_variance):
    X_har = pd.DataFrame({
        "RV_D": real_variance,                   # yesterday’s daily RV
        "RV_W": real_variance.rolling(5).mean(), # weekly avg of daily RVs
        "RV_M": real_variance.rolling(21).mean() # monthly avg of daily RVs
    })
    return X_har


def create_iv_surface_predictors(  
    options, 
    iv_surface_model, 
    params=None, 
    r=0.0, 
    q=0.0
):
    iv_features = []
    T_30 = 30 / 252
    T_60 = 60 / 252

    for date, chain in options.groupby("date"):
        # --- Underlying spot (ATM anchor) ---
        S = float(chain["underlying_last"].iloc[0])

        # --- Fit or restore the surface ---
        if params is None or date not in params:
            iv_surface_model.fit(chain)
        else:
            iv_surface_model.set_params({**params[date], "spot": S})

        # --- ATM IVs ---
        atm_iv_30d = iv_surface_model.implied_vol(S, T_30)
        atm_iv_60d = iv_surface_model.implied_vol(S, T_60)

        # --- 25Δ strikes (approx using ATM vol for the delta inversion) ---
        # target deltas: put = -0.25, call = +0.25
        K_put_25d = solve_strike_for_delta(
            target_delta=-0.25,
            S=S,
            T=T_30,
            sigma=atm_iv_30d,
            option_type="put",
            r=r,
            q=q
        )
        K_call_25d = solve_strike_for_delta(
            target_delta=0.25,
            S=S,
            T=T_30,
            sigma=atm_iv_30d,
            option_type="call",
            r=r,
            q=q
        )

        # --- 25Δ skew (downside - upside) ---
        iv_put_25d = iv_surface_model.implied_vol(K_put_25d, T_30)
        iv_call_25d = iv_surface_model.implied_vol(K_call_25d, T_30)
        iv_skew = iv_put_25d - iv_call_25d

        # --- Term structure slope (30D → 60D) ---
        iv_ts = atm_iv_60d - atm_iv_30d

        iv_features.append({
            "date": date,
            "atm_iv_30d": atm_iv_30d,
            "iv_skew": iv_skew,
            "iv_ts": iv_ts,
        })

    iv_features = pd.DataFrame(iv_features).set_index("date").sort_index()
    return iv_features


def intraday_log_returns(prices: pd.Series) -> pd.Series:
    """Simple helper: log-returns of an intraday price series."""
    return np.log(prices).diff()


def overnight_return(df, rth_start="09:30", rth_end="16:00"):
    """
    df: 5-min ES OHLCV with DatetimeIndex (exchange timezone)
    r_ov_t = log( Open_RTH_t / Close_RTH_{t-1} )
    """
    tod = df.index.strftime("%H:%M")
    rth = df[(tod >= rth_start) & (tod <= rth_end)].copy()

    # RTH open = first 'open' of the day, RTH close = last 'close'
    daily_open  = rth.groupby(rth.index.date)["open"].first()
    daily_close = rth.groupby(rth.index.date)["close"].last()

    daily_open.index  = pd.to_datetime(daily_open.index)
    daily_close.index = pd.to_datetime(daily_close.index)

    ov = np.log(daily_open / daily_close.shift(1))
    ov.name = "overnight_ret"
    return ov.dropna()


def realized_skew_kurt_intraday(
        intraday_returns: pd.Series
    ) -> pd.DataFrame:
    # group by calendar day
    g = intraday_returns.groupby(intraday_returns.index.date)

    out = []
    for day, r in g:
        r = r.dropna().to_numpy()
        N = len(r)
        if N == 0:
            rsk = np.nan
            rku = np.nan
        else:
            s2 = np.sum(r**2)
            s3 = np.sum(r**3)
            s4 = np.sum(r**4)

            if s2 <= 0:
                rsk = np.nan
                rku = np.nan
            else:
                rsk = np.sqrt(N) * s3 / (s2 ** 1.5)   # RSK_t
                rku = N * s4 / (s2 ** 2)             # RKU_t

        out.append((pd.to_datetime(day), rsk, rku))

    df = pd.DataFrame(out, columns=["date", "rsk", "rku"]).set_index("date").sort_index()
    return df


def create_return_predictors(
        daily_returns: pd.Series, 
        intraday_prices: pd.Series, 
        h: int = 21, 
        rth_start="09:30", 
        rth_end="16:00"
    ) -> pd.DataFrame:
    ret_features = pd.DataFrame(index=daily_returns.index)

    # ---------- 1) Overnight jumps (RTH close -> next RTH open) ----------
    ov = overnight_return(intraday_prices, rth_start=rth_start, rth_end=rth_end)
    ret_features["overnight_ret"] = ov.reindex(ret_features.index).rolling(5).mean()

    # ---------- 2) Volatility clustering from daily returns ----------
    ret_features["abs_r"] = daily_returns.abs().rolling(5).mean()
    ret_features["r2"] = daily_returns.pow(2).rolling(5).mean()

    # ---------- 3) Asymmetry / leverage ----------
    is_neg_ret = (daily_returns < 0).astype(int)
    ret_features["neg_r2"] = (is_neg_ret * daily_returns.pow(2)).rolling(5).mean()

    # ---------- 4) Downside / upside semivariance over horizon h ----------
    rolling = daily_returns.rolling(h)
    ret_features["down_var"] = rolling.apply(
        lambda x: (np.minimum(x, 0.0) ** 2).mean(), raw=False
    )
    ret_features["up_var"] = rolling.apply(
        lambda x: (np.maximum(x, 0.0) ** 2).mean(), raw=False
    )

    # ---------- 5) Realized skewness & kurtosis from intraday RTH ----------
    tod = intraday_prices.index.strftime("%H:%M")
    rth_prices = intraday_prices.loc[(tod >= rth_start) & (tod <= rth_end), "close"]
    intraday_returns = intraday_log_returns(rth_prices)
    daily_moments = realized_skew_kurt_intraday(intraday_returns)
    daily_roll_moments = daily_moments.rolling(21).mean()
    ret_features = ret_features.join(daily_roll_moments, how="left")

    return ret_features


def create_market_features(start: str, end: str) -> pd.DataFrame:
    tickers = ["^VIX", "^VVIX", "^VIX3M"]
    vix = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[["Close"]]
    vix.columns = [tkr.replace("^", "") for _, tkr in vix.columns]
    #vix = vix.rename(columns={"VIX": "VIX", "VVIX": "VVIX"}).sort_index()
    vix["vix_ts"] = vix["VIX3M"] - vix["VIX"]
    vix = vix.drop("VIX3M", axis=1)
    return vix


def feature_engineering(
    X_core: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 21,
    ewma_alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Create X_eng features from economic predictors.
    Returns ONLY the X_eng features (not X_core itself).
    """
    X = X_core.copy()
    X_eng = {}

    # -------------------------------------------------
    # 1. Horizon-aligned smoothing (regime level)
    # -------------------------------------------------
    if "VIX" in X:
        X_eng["VIX_rm5"]   = X["VIX"].rolling(window_short, min_periods=1).mean()
        X_eng["VIX_rm21"]  = X["VIX"].rolling(window_long, min_periods=1).mean()
        X_eng["VIX_ewma"]  = X["VIX"].ewm(alpha=ewma_alpha, adjust=False).mean()

    if "RV_D" in X:
        X_eng["RV_D_ewma"] = X["RV_D"].ewm(alpha=ewma_alpha, adjust=False).mean()

    if "HY_OAS" in X:
        X_eng["HY_OAS_ewma"] = X["HY_OAS"].ewm(alpha=ewma_alpha, adjust=False).mean()

    # -------------------------------------------------
    # 2. Regime shift / stress dynamics
    # -------------------------------------------------
    if "VIX" in X:
        X_eng["dVIX_5d"] = X["VIX"] - X["VIX"].shift(window_short)

    if "iv_skew" in X:
        X_eng["dSkew_5d"] = X["iv_skew"] - X["iv_skew"].shift(window_short)

    if "atm_iv_30d" in X and "RV_M" in X:
        X_eng["iv_minus_realized"] = X["atm_iv_30d"] - X["RV_M"]

    if "VVIX" in X and "VIX" in X:
        X_eng["vvix_over_vix"] = X["VVIX"] / X["VIX"].replace(0, np.nan)

    if "VIX" in X and "HY_OAS" in X:
        X_eng["VIX_time_HY_OAS"] = X["VIX"] * X["HY_OAS"] 
    
    if "RV_D" in X:
        X_eng["RV_D_rollvol5"] = X["RV_D"].rolling(5).std()
        X_eng["RV_D_rollvol21"] = X["RV_D"].rolling(21).std()

    # -------------------------------------------------
    # 3. Assemble into DataFrame
    # -------------------------------------------------
    X_eng = pd.DataFrame(X_eng, index=X.index)

    return X_eng


def build_naive_targets(
    rv_m: pd.Series,
    iv_atm: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Build naive benchmarks on the *same index* as rv_m:

      - Naive RV: log(monthly realized variance RV_M)
      - Naive IV: log( (ATM 30D IV^2) / 252 )   [optional]

    Parameters
    ----------
    rv_m : pd.Series
        Monthly realized variance (already on variance scale).
    iv_atm : pd.Series or None, optional
        ATM 30D implied vol (annualised). If None, only Naive RV is built.

    Returns
    -------
    y_naive_rv : pd.Series
        log(RV_M) on rv_m's index.
    y_naive_iv : pd.Series or None
        log( (IV_30D^2)/252 ) aligned to rv_m, or None if iv_atm is None.
    """
    # ensure Series and drop any obvious NaNs
    rv_m = rv_m.squeeze().dropna()

    # --- Naive RV: monthly realized variance, already on variance scale ---
    y_naive_rv = np.log(rv_m)
    y_naive_rv.name = "log_fwd_var_naive_rv"

    # --- No IV case: RV-only benchmark ---
    if iv_atm is None:
        return y_naive_rv, None

    # --- IV case: align indices and build IV-based benchmark ---
    iv_atm = iv_atm.squeeze().dropna()

    # align to the common index (safety)
    idx = rv_m.index.intersection(iv_atm.index)
    rv_m_aligned = rv_m.loc[idx]
    iv_atm_aligned = iv_atm.loc[idx]

    # rebuild RV naive on aligned index (so both outputs line up)
    y_naive_rv = np.log(rv_m_aligned)
    y_naive_rv.name = "log_fwd_var_naive_rv"

    # Naive IV: ATM 30D implied vol (annualised) -> daily variance -> log
    var_daily_iv = (iv_atm_aligned ** 2) / 252.0
    y_naive_iv = np.log(var_daily_iv)
    y_naive_iv.name = "log_fwd_var_naive_iv"

    return y_naive_rv, y_naive_iv


def build_har_vix_dataset(es_5min, start=None, end=None, h=21):
    # 1) daily RV
    daily_rv = rv_intraday(es_5min["close"])

    if start is not None:
        daily_rv = daily_rv.loc[start:]
    if end is not None:
        daily_rv = daily_rv.loc[:end]

    # 2) target
    y = create_forward_target(daily_rv, horizon=h)

    # 3) predictors
    X_har = create_har_lags(daily_rv)   # RV_D, RV_W, RV_M, ...

    # use the target index to get the proper VIX range
    feat_start = y.index.min()
    feat_end   = y.index.max()

    X_vix = create_market_features(start=feat_start, end=feat_end)[["VIX"]]
    X_vix = X_vix.reindex(y.index).ffill()

    X = pd.concat([X_har, X_vix], axis=1)

    # 4) align and drop NaNs
    data = pd.concat([X, y], axis=1).dropna()
    X_final = data[X.columns]
    y_final = data[y.name]

    return X_final, y_final