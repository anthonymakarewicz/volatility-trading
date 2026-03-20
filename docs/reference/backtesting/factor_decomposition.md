# Options Engine Factor Decomposition

This page documents the explicit factor-decomposition models available in the
options backtesting engine.

The goal is to explain **why** a strategy made or lost money in factor terms,
without pretending to be a full volatility-surface risk system.

## Current Scope

The engine currently supports one explicit factor model:

- `RiskReversalFactorModel`

This model is intended for a **same-expiry risk reversal**:

- one put leg
- one call leg
- shared expiry group

The built-in `skew_mispricing` preset attaches this model automatically.

Custom strategies can opt in by attaching a factor model directly to
`StrategySpec.factor_decomposition_model`.

## Business Logic

A same-expiry risk reversal is not mainly a "parallel vol" trade. It is a
combination of:

- **vol level** exposure: both wings move together
- **risk-reversal / skew** exposure: call wing and put wing move differently

If reporting only shows:

- total `vega`
- one scalar `iv`

then the explanation is too compressed for a skew strategy. You can see that
the book had vol exposure, but not whether P&L came from:

- overall vol moving, or
- skew steepening / flattening

The RR factor model splits that vol contribution into:

- `iv_level`
- `rr_skew`

## Signed Leg Vega

The decomposition uses **position-signed** leg vega, not raw quote vega.

For each leg:

```text
signed_vega =
    effective_leg_side
    * quote.vega
    * leg_units
    * option_contract_multiplier
    * contracts
```

Interpretation:

- long leg -> positive contribution
- short leg -> negative contribution
- ratio weights, contract multiplier, and strategy contract count are included

For a long risk reversal:

- long call -> positive call vega
- short put -> negative put vega

## Factor Definitions

Let:

- `call_iv` = call-leg implied volatility in **vol points**
- `put_iv` = put-leg implied volatility in **vol points**
- `call_vega` = signed position call vega
- `put_vega` = signed position put vega

The RR factor model defines:

```text
iv_level = 0.5 * (call_iv + put_iv)
rr_skew = call_iv - put_iv
```

and factor exposures:

```text
factor_exposure_iv_level = call_vega + put_vega
factor_exposure_rr_skew = 0.5 * (call_vega - put_vega)
```

Interpretation:

- `iv_level` is the average wing IV level
- `rr_skew` is the call-minus-put risk reversal level
- `factor_exposure_iv_level` is the net parallel-vol exposure
- `factor_exposure_rr_skew` is the net skew exposure

## Why The `1/2` Appears

The `0.5` terms are not arbitrary.

They are chosen so the compressed factor P&L is exactly the same linear vega
P&L as the two-leg call/put representation.

Start from:

```text
d_iv_level = 0.5 * (d_call_iv + d_put_iv)
d_rr_skew = d_call_iv - d_put_iv
```

Now define:

```text
E_level = call_vega + put_vega
E_rr = 0.5 * (call_vega - put_vega)
```

Then:

```text
E_level * d_iv_level + E_rr * d_rr_skew
```

expands to:

```text
call_vega * d_call_iv + put_vega * d_put_iv
```

So the factor decomposition is a **change of basis** for the linear vega P&L.

That is the key result:

- the implementation does not invent a new approximate P&L number
- it rewrites the same linear call/put vol P&L in more interpretable RR terms

## Daily P&L Attribution

The dense daily attribution table uses prior-day factor exposure times
today's factor move:

```text
IV_Level_PnL = factor_exposure_iv_level_prev * d_factor_iv_level
RR_Skew_PnL = factor_exposure_rr_skew_prev * d_factor_rr_skew
```

When factor columns are present, the daily table also sets:

```text
Vega_PnL = IV_Level_PnL + RR_Skew_PnL + ...
```

So:

- `Vega_PnL` remains the aggregate vol contribution
- factor columns provide the more detailed split underneath

## Persisted Outputs

When a strategy attaches a factor model, the standard workflow/reporting path
persists factor information into:

- `exposures_daily.csv`
  - `factor_exposure_iv_level`
  - `factor_exposure_rr_skew`
- `pnl_attribution_daily.csv`
  - `IV_Level_PnL`
  - `RR_Skew_PnL`

The naming is dynamic at the engine level, but those are the concrete public
columns for the current RR model.

Current factor **state** values such as:

- `factor_iv_level`
- `factor_rr_skew`

remain available on the dense daily MTM path, but are not exported in the
public `exposures_daily.csv` artifact because they are factor states rather
than exposures.

## What This Does Not Do

This feature is intentionally narrower than a full volatility-surface risk
engine.

It does **not** currently model:

- term structure factors
- curvature / fly factors
- node-by-node or bucketed surface vegas
- vol convexity / vomma attribution
- portfolio-level multi-underlying surface aggregation
- stress-scenario sizing for skew steepening / flattening

Those belong to later risk-model or portfolio-engine work.

## Implementation Boundary

The factor-model choice is attached at the strategy/spec level, not run-wide
`ModelingConfig`.

Reason:

- factor decomposition is **structure-aware**
- different structures need different explanatory models
- a run-wide global factor model would be the wrong abstraction once the engine
  supports multiple strategy types in the same run

So the split is:

- `StrategySpec.factor_decomposition_model`
  - selects **which** explanatory model applies
- run/modeling configuration
  - can later tune calibration details, but should not choose the structure
    semantics itself

## Recommended Usage

Use factor decomposition when you want to answer questions like:

- Did this skew strategy make money because overall vol fell?
- Or because skew flattened?
- Was the book carrying mostly net parallel-vol risk or mostly RR/skew risk?

Do not treat it as a replacement for:

- scenario-based sizing
- bucket risk reports
- full portfolio-level surface controls

It is an explanatory layer, not the entire risk system.
