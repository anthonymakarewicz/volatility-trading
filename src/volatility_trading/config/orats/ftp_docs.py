STRIKES_VENDOR_COL_DOCS = {
    "ticker": "Underlying symbol representing the stock or index.",
    "cOpra": (
        "Full OCC/OPRA symbol for the call contract (present from ~2010 onward)"
        "(e.g. 'SPXW140118C01375000')."
    ),
    "pOpra": (
        "Full OCC/OPRA symbol for the put contract (present from ~2010 onward)"
        "(e.g. 'SPXW140118P01375000')."
    ),
    "stkPx": (
        "Current price of the underlying stock at the time of the "
        "options snapshot capture (14 minutes before the close)."
        "For indexes, this is the solved implied futures price "
        "using put-call parity for each expiration."
    ),
    "expirDate": "Calendar date on which the option expires (MM/DD/YYYY).",
    "yte": "Time to expiration expressed in years.",
    "strike": "Strike price at which the option can be exercised.",

    "cVolu": "Total number of call contracts traded on the quote date.",
    "pVolu": "Total number of put contracts traded on the quote date.",
    "cOi": "Total outstanding call open interest reported by OCC the night before.",
    "pOi": "Total outstanding put open interest reported by OCC the night before.",

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

    "extVol": "External implied volatility for the underlying, from the ORATS forecast volatility.",
    "extCTheo": "External theoretical value of the call, from a third-party source.",
    "extPTheo": "External theoretical value of the put, from a third-party source.",

    "spot_px": (
        "The current market price of the underlying asset."
        "For indexes this is the cash price."
    ),
    "trade_date": "Date on which the option was traded / quoted (as string MM/DD/YYYY).",
}


STRIKES_CANONICAL_COL_DOCS = {
    # New / derived stuff (worth documenting explicitly)
    "dte": "Days to expiry: expiry_date - trade_date in calendar days.",
    "moneyness_ks": "Strike divided by spot_price (K / S), dimensionless moneyness.",

    "call_mid_price": "Mid price for the call: (call_bid + call_ask) / 2.",
    "put_mid_price": "Mid price for the put: (put_bid + put_ask) / 2.",

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