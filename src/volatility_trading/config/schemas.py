from __future__ import annotations

import polars as pl

# --- ORATS SMV Strikes schema (Near-EOD) ------------------------------------ #
# Dtypes are chosen to be robust for ETL:
# - prices/greeks/rates/IVs  -> Float64
# - volumes / open interest  -> Int64
# - dates                    -> Utf8 (parsed to Date later in cleaning)
# - tickers                  -> Utf8

ORATS_DTYPE = {
    "ticker": pl.Utf8,          # underlying symbol (stock or index)
    "stkPx": pl.Float64,        # current price of the underlying (spot or implied future)
    "expirDate": pl.Utf8,       # option expiration date (MM/DD/YYYY)
    "yte": pl.Float64,          # years to expiration
    "strike": pl.Float64,       # option strike price

    "cVolu": pl.Int64,          # total call volume for the day
    "cOi": pl.Int64,            # call open interest (OCC, previous night)
    "pVolu": pl.Int64,          # total put volume for the day
    "pOi": pl.Int64,            # put open interest (OCC, previous night)

    "cBidPx": pl.Float64,       # NBBO bid price for the call
    "cValue": pl.Float64,       # theoretical call value from smooth vol model
    "cAskPx": pl.Float64,       # NBBO ask price for the call

    "pBidPx": pl.Float64,       # NBBO bid price for the put
    "pValue": pl.Float64,       # theoretical put value from smooth vol model
    "pAskPx": pl.Float64,       # NBBO ask price for the put

    "cBidIv": pl.Float64,       # call IV at NBBO bid
    "cMidIv": pl.Float64,       # call IV at NBBO mid
    "cAskIv": pl.Float64,       # call IV at NBBO ask
    "smoothSmvVol": pl.Float64, # smoothed IV from ORATS surface

    "pBidIv": pl.Float64,       # put IV at NBBO bid
    "pMidIv": pl.Float64,       # put IV at NBBO mid
    "pAskIv": pl.Float64,       # put IV at NBBO ask

    "iRate": pl.Float64,        # continuously-compounded risk-free rate
    "divRate": pl.Float64,      # continuous dividend yield of discrete dividends
    "residualRateData": pl.Float64,  # residual rate implied by pricing model

    "delta": pl.Float64,        # option delta
    "gamma": pl.Float64,        # option gamma
    "theta": pl.Float64,        # option theta (1-day decay)
    "vega": pl.Float64,         # sensitivity to 1-pt change in implied vol
    "rho": pl.Float64,          # sensitivity to 1% change in rates
    "phi": pl.Float64,          # convexity measure wrt underlying price
    "driftlessTheta": pl.Float64,  # theta excluding drift in underlying

    "extVol": pl.Float64,       # external implied vol from third-party source
    "extCTheo": pl.Float64,     # external theoretical call value
    "extPTheo": pl.Float64,     # external theoretical put value

    "spot_px": pl.Float64,      # current spot price (cash index / underlying)
    "trade_date": pl.Utf8,      # trade date of the option quote (MM/DD/YYYY)
}

# Optional: human-readable data dictionary
ORATS_COLUMN_DOCS = {
    "ticker": "Underlying symbol representing the stock or index.",
    "stkPx": "Current price of the underlying stock. For indexes, this is the solved implied futures price for each expiration.",
    "expirDate": "Calendar date on which the option expires.",
    "yte": "Time to expiration expressed in years.",
    "strike": "Strike price at which the option can be exercised.",

    "cVolu": "Total number of call contracts traded on the quote date.",
    "cOi": "Total outstanding call open interest reported by OCC.",
    "pVolu": "Total number of put contracts traded on the quote date.",
    "pOi": "Total outstanding put open interest reported by OCC.",

    "cBidPx": "NBBO bid price at which a market maker is willing to buy the call.",
    "cValue": "Theoretical call value based on a smooth volatility assumption.",
    "cAskPx": "NBBO ask price at which a market maker is willing to sell the call.",

    "pBidPx": "NBBO bid price at which a market maker is willing to buy the put.",
    "pValue": "Theoretical put value based on a smooth volatility assumption.",
    "pAskPx": "NBBO ask price at which a market maker is willing to sell the put.",

    "cBidIv": "Implied volatility of the call at the NBBO bid price.",
    "cMidIv": "Implied volatility of the call at the NBBO mid price.",
    "cAskIv": "Implied volatility of the call at the NBBO ask price.",
    "smoothSmvVol": "Smoothed implied volatility from the ORATS model.",

    "pBidIv": "Implied volatility of the put at the NBBO bid price.",
    "pMidIv": "Implied volatility of the put at the NBBO mid price.",
    "pAskIv": "Implied volatility of the put at the NBBO ask price.",

    "iRate": "Continuously-compounded risk-free interest rate.",
    "divRate": "Continuous dividend yield implied by discrete dividends.",
    "residualRateData": "Residual interest rate inferred from the option pricing model.",

    "delta": "Change in option price for a one-unit change in underlying price.",
    "gamma": "Change in delta for a one-unit change in underlying price.",
    "theta": "One-day time decay of the option's value.",
    "vega": "Change in option price for a one-point change in implied volatility.",
    "rho": "Change in option price for a one-percent change in interest rates.",
    "phi": "Convexity measure of the option price with respect to underlying price.",
    "driftlessTheta": "Theta of the option ignoring drift in the underlying price.",

    "extVol": "External implied volatility for the underlying, from a third-party source.",
    "extCTheo": "External theoretical value of the call, from a third-party source.",
    "extPTheo": "External theoretical value of the put, from a third-party source.",

    "spot_px": "Current market (cash) price of the underlying asset.",
    "trade_date": "Date on which the option was traded / quoted.",
}