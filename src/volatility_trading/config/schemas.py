from __future__ import annotations

import polars as pl

# --- ORATS SMV Strikes schema (Near-EOD) ------------------------------------ #
# Dtypes are chosen to be robust for ETL:
# - prices/greeks/rates/IVs  -> Float64
# - volumes / open interest  -> Int64
# - dates                    -> Utf8 (parsed to Date later in cleaning)
# - tickers                  -> Utf8

ORATS_DTYPE = {
    "ticker": pl.Utf8,
    "stkPx": pl.Float64,
    "expirDate": pl.Utf8,
    "yte": pl.Float64,
    "strike": pl.Float64,
    "cVolu": pl.Int64,
    "cOi": pl.Int64,
    "pVolu": pl.Int64,
    "pOi": pl.Int64,
    "cBidPx": pl.Float64,
    "cValue": pl.Float64,
    "cAskPx": pl.Float64,
    "pBidPx": pl.Float64,
    "pValue": pl.Float64,
    "pAskPx": pl.Float64,
    "cBidIv": pl.Float64,
    "cMidIv": pl.Float64,
    "cAskIv": pl.Float64,
    "smoothSmvVol": pl.Float64,
    "pBidIv": pl.Float64,
    "pMidIv": pl.Float64,
    "pAskIv": pl.Float64,
    "iRate": pl.Float64,
    "divRate": pl.Float64,
    "residualRateData": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "theta": pl.Float64,
    "vega": pl.Float64,
    "rho": pl.Float64,
    "phi": pl.Float64,
    "driftlessTheta": pl.Float64,
    "extVol": pl.Float64,
    "extCTheo": pl.Float64,
    "extPTheo": pl.Float64,
    "spot_px": pl.Float64,
    "trade_date": pl.Utf8,
}

ORATS_COLUMN_DOCS = {
    "ticker": "Underlying symbol representing the stock or index.",
    "stkPx": (
        "Current price of the underlying stock. For indexes, this may be the "
        "implied futures/forward price used by ORATS for that expiry."
    ),
    "expirDate": "Calendar date on which the option expires (MM/DD/YYYY).",
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
    "smoothSmvVol": "Smoothed implied volatility from the ORATS surface model.",

    "pBidIv": "Implied volatility of the put at the NBBO bid price.",
    "pMidIv": "Implied volatility of the put at the NBBO mid price.",
    "pAskIv": "Implied volatility of the put at the NBBO ask price.",

    "iRate": "Continuously-compounded risk-free interest rate.",
    "divRate": "Continuous dividend yield implied by discrete dividends / borrow.",
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
    "trade_date": "Date on which the option was traded / quoted (as string MM/DD/YYYY).",
}


# --- ORATS processed WIDE panel schema ------------------------------------- #
# This describes the *processed* per-ticker panel written by build_orats_panel_for_ticker.

ORATS_VENDOR_TO_PROCESSED = {
    # identifiers
    "ticker": "ticker",

    # underlying / dates
    "stkPx": "spot_price",          # main underlying price used in ORATS model (can be fwd for indexes)
    "spot_px": "spot_price_cash",   # cash/spot index or underlying
    "expirDate": "expiry_date",     # will be parsed to Date
    "trade_date": "trade_date",     # will be parsed to Date
    "yte": "yte",
    "strike": "strike",

    # volume / open interest
    "cVolu": "call_volume",
    "cOi": "call_open_interest",
    "pVolu": "put_volume",
    "pOi": "put_open_interest",

    # quotes (prices)
    "cBidPx": "call_bid",
    "cValue": "call_theo",
    "cAskPx": "call_ask",

    "pBidPx": "put_bid",
    "pValue": "put_theo",
    "pAskPx": "put_ask",

    # implied vols
    "cBidIv": "call_bid_iv",
    "cMidIv": "call_mid_iv",
    "cAskIv": "call_ask_iv",
    "smoothSmvVol": "smoothed_vol",

    "pBidIv": "put_bid_iv",
    "pMidIv": "put_mid_iv",
    "pAskIv": "put_ask_iv",

    # rates
    "iRate": "risk_free_rate",
    "divRate": "dividend_yield",
    "residualRateData": "residual_rate_data",

    # greeks (raw from ORATS)
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    "rho": "rho",
    "phi": "phi",
    "driftlessTheta": "driftless_theta",

    # external vols / theo values
    "extVol": "ext_vol",
    "extCTheo": "ext_call_theo",
    "extPTheo": "ext_put_theo",
}

ORATS_PROCESSED_COLUMN_DOCS = {
    # Direct renames (reuse raw docs)
    "ticker": ORATS_COLUMN_DOCS["ticker"],
    "trade_date": ORATS_COLUMN_DOCS["trade_date"] + " Parsed to a Date.",
    "expiry_date": ORATS_COLUMN_DOCS["expirDate"] + " Parsed to a Date.",
    "yte": ORATS_COLUMN_DOCS["yte"],
    "spot_price": ORATS_COLUMN_DOCS["stkPx"],
    "strike": ORATS_COLUMN_DOCS["strike"],
    "call_volume": ORATS_COLUMN_DOCS["cVolu"],
    "put_volume": ORATS_COLUMN_DOCS["pVolu"],
    "call_bid": ORATS_COLUMN_DOCS["cBidPx"],
    "call_ask": ORATS_COLUMN_DOCS["cAskPx"],
    "put_bid": ORATS_COLUMN_DOCS["pBidPx"],
    "put_ask": ORATS_COLUMN_DOCS["pAskPx"],
    "call_bid_iv": ORATS_COLUMN_DOCS["cBidIv"],
    "call_mid_iv": ORATS_COLUMN_DOCS["cMidIv"],
    "call_ask_iv": ORATS_COLUMN_DOCS["cAskIv"],
    "put_bid_iv": ORATS_COLUMN_DOCS["pBidIv"],
    "put_mid_iv": ORATS_COLUMN_DOCS["pMidIv"],
    "put_ask_iv": ORATS_COLUMN_DOCS["pAskIv"],
    "smoothed_vol": ORATS_COLUMN_DOCS["smoothSmvVol"],
    "risk_free_rate": ORATS_COLUMN_DOCS["iRate"],
    "dividend_yield": ORATS_COLUMN_DOCS["divRate"],
    "ext_vol": ORATS_COLUMN_DOCS["extVol"],

    # Call greeks: also basically direct renames
    "call_delta": ORATS_COLUMN_DOCS["delta"],
    "call_gamma": ORATS_COLUMN_DOCS["gamma"],
    "call_theta": ORATS_COLUMN_DOCS["theta"],
    "call_vega": ORATS_COLUMN_DOCS["vega"],
    "call_rho": ORATS_COLUMN_DOCS["rho"],

    # New / derived stuff (worth documenting explicitly)
    "dte": "Days to expiry: expiry_date - trade_date in calendar days.",
    "moneyness_ks": "Strike divided by spot_price (K / S), dimensionless moneyness.",

    "call_mid": "Mid price for the call: (call_bid + call_ask) / 2.",
    "put_mid": "Mid price for the put: (put_bid + put_ask) / 2.",

    "call_spread": "Call bid–ask spread in price units: call_ask - call_bid.",
    "put_spread": "Put bid–ask spread in price units: put_ask - put_bid.",

    "call_rel_spread": (
        "Call bid–ask spread as a fraction of call_mid, used as a liquidity "
        "proxy (None when mid <= 0 or spread < 0)."
    ),
    "put_rel_spread": (
        "Put bid–ask spread as a fraction of put_mid, used as a liquidity "
        "proxy (None when mid <= 0 or spread < 0)."
    ),

    # Put Greeks (parity-based, deserve explicit explanation)
    "put_delta": (
        "Put delta reconstructed via European put–call parity, using call_delta, "
        "risk_free_rate, dividend_yield and yte (BS-style model with continuous "
        "dividend yield)."
    ),
    "put_gamma": (
        "Put gamma set equal to call_gamma via put–call parity (same gamma for "
        "calls and puts under BS-style model)."
    ),
    "put_theta": (
        "Put theta reconstructed via put–call parity, consistent with call_theta, "
        "risk_free_rate, dividend_yield and yte."
    ),
    "put_vega": (
        "Put vega set equal to call_vega via put–call parity (same vega under "
        "BS-style model)."
    ),
    "put_rho": (
        "Put rho reconstructed via put–call parity, consistent with call_rho and "
        "the forward/discounting used by ORATS."
    ),
}

# Core processed WIDE schema (the default selection in build_orats_panel_for_ticker)
CORE_ORATS_WIDE_COLUMNS = [
    "ticker",
    "trade_date",
    "expiry_date",
    "dte",
    "yte",
    "spot_price",
    "strike",
    "moneyness_ks",
    "call_volume",
    "put_volume",
    "call_bid",
    "call_ask",
    "call_mid",
    "put_bid",
    "put_ask",
    "put_mid",
    "call_spread",
    "put_spread",
    "call_rel_spread",
    "put_rel_spread",
    "call_bid_iv",
    "call_mid_iv",
    "call_ask_iv",
    "put_bid_iv",
    "put_mid_iv",
    "put_ask_iv",
    "smoothed_vol",
    "risk_free_rate",
    "dividend_yield",
    "ext_vol",
    # call Greeks (from ORATS)
    "call_delta",
    "call_gamma",
    "call_theta",
    "call_vega",
    "call_rho",
    # put Greeks (via parity)
    "put_delta",
    "put_gamma",
    "put_theta",
    "put_vega",
    "put_rho",
]

# Minimal set of columns that must always be present in any processed ORATS panel
REQUIRED_ORATS_WIDE_COLUMNS = {
    "ticker",
    "trade_date",
    "expiry_date",
    "dte",
    "spot_price",
    "strike",
}