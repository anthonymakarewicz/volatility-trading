# Which OPRA root we consider the "canonical" product for a given ticker.

# For SPX, we want the PM-settled weeklies / modern chain: SPXW.
PREFERRED_OPRA_ROOT: dict[str, str] = {
    "SPX": "SPXW",
    # in the future we could add e.g. "NDX": "NDXP", etc.
}