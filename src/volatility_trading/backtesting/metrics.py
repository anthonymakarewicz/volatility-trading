import pandas as pd

def to_daily_mtm(raw_mtm: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Take raw MTM (trade-level or event-level) and produce a daily P&L
    and Greek-based attribution.

    Assumptions:
    - df["iv"] is implied vol in DECIMALS (e.g. 0.16 for 16%).
    - df["vega"] is position vega in $ PER 1 VOL POINT (i.e. per +1% IV).
    - df["theta"] is position theta in $ PER DAY.
    """
    df = raw_mtm.copy()
    df.index = pd.to_datetime(df.index)

    # 1) Reindex to calendar days
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full)

    # 2) Fill P&L and carry-forward Greeks, spot & iv
    df["delta_pnl"] = df["delta_pnl"].fillna(0.0)
    for g in ["net_delta", "delta", "gamma", "vega", "theta", "S", "iv"]:
        df[g] = df[g].ffill().fillna(0.0)

    # 3) Compute daily moves
    df["dS"] = df["S"].diff().fillna(0.0)

    # IV in vol points (1 point = 1% absolute IV)
    df["iv_pts"]   = df["iv"] * 100.0              # 0.16 -> 16.0
    df["d_iv_pts"] = df["iv_pts"].diff().fillna(0.0)

    # Calendar-day step (for theta per day)
    df["dt"] = df.index.to_series().diff().dt.days.fillna(1).astype(int)

    # 4) Shift prior-day Greeks
    df["net_delta_prev"] = df["net_delta"].shift(1).fillna(0.0)
    df["delta_prev"]     = df["delta"].shift(1).fillna(0.0)
    df["gamma_prev"]     = df["gamma"].shift(1).fillna(0.0)
    df["vega_prev"]      = df["vega"].shift(1).fillna(0.0)    # $ / vol point
    df["theta_prev"]     = df["theta"].shift(1).fillna(0.0)   # $ / day

    # 5) Greek P&L attribution
    df["Delta_PnL"]          = df["net_delta_prev"] * df["dS"]
    df["Unhedged_Delta_PnL"] = df["delta_prev"]     * df["dS"]
    df["Gamma_PnL"]          = 0.5 * df["gamma_prev"] * df["dS"]**2

    # Vega: vega_prev is $ / vol point, d_iv_pts is change in vol points
    df["Vega_PnL"]           = df["vega_prev"] * df["d_iv_pts"]

    # Theta: theta_prev is $ / day, dt is number of days passed
    df["Theta_PnL"]          = df["theta_prev"] * df["dt"]

    # 6) Residual
    df["Other_PnL"] = df["delta_pnl"] - (
        df["Delta_PnL"]
        + df["Gamma_PnL"]
        + df["Vega_PnL"]
        + df["Theta_PnL"]
    )

    # 7) Equity curve
    df["equity"] = initial_capital + df["delta_pnl"].cumsum()
    return df