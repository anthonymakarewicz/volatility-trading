import pandas as pd

def to_daily_mtm(raw_mtm: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    df = raw_mtm.copy()
    df.index = pd.to_datetime(df.index)

    # reindex to calendar days
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full)

    # fill P&L and carry-forward Greeks, spot & iv
    df["delta_pnl"] = df["delta_pnl"].fillna(0.0)
    for g in ["net_delta", "delta", "gamma", "vega", "theta", "S", "iv"]:
        df[g] = df[g].ffill().fillna(0.0)

    # compute daily moves
    df["dS"]     = df["S"].diff().fillna(0.0)
    df["dsigma"] = df["iv"].diff().fillna(0.0)
    df["dt"]     = df.index.to_series().diff().dt.days.fillna(1).astype(int)

    # shift prior‐day Greeks
    df["net_delta_prev"] = df["net_delta"].shift(1).fillna(0.0)
    df["delta_prev"]     = df["delta"].shift(1).fillna(0.0)
    df["gamma_prev"]     = df["gamma"].shift(1).fillna(0.0)
    df["vega_prev"]      = df["vega"].shift(1).fillna(0.0)
    df["theta_prev"]     = df["theta"].shift(1).fillna(0.0)

    # greek PnL attribution
    df["Delta_PnL"]          = df["net_delta_prev"] * df["dS"]
    df["Unhedged_Delta_PnL"] = df["delta_prev"] * df["dS"]
    df["Gamma_PnL"]          = 0.5 * df["gamma_prev"] * df["dS"]**2
    df["Vega_PnL"]           = df["vega_prev"] * df["dsigma"]
    df["Theta_PnL"]          = df["theta_prev"] * df["dt"]

    df["Other_PnL"] = df["delta_pnl"] - (
        df["Delta_PnL"]
        + df["Gamma_PnL"]
        + df["Vega_PnL"]
        + df["Theta_PnL"]
    )

    df["equity"] = initial_capital + df["delta_pnl"].cumsum()
    return df