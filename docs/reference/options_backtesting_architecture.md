# Options Backtesting Architecture

This document describes how the current options backtesting stack is wired, from
`Backtester` orchestration to strategy-specific presets such as VRP harvesting.

## Scope

- Covered: options strategy runtime (`backtesting/` + `strategies/options_core/`)
- Not covered: data ETL/QC pipelines and model research notebooks

## Design Intent

The architecture separates:

1. **Backtest orchestration** (engine, runtime context, strategy contract)
2. **Generic options execution runtime** (entry selection, sizing, lifecycle)
3. **Strategy presets** (VRP or future skew/IV-RV variants)

This keeps strategy-specific code minimal while reusing the same lifecycle and
accounting logic across structures (single-leg or multi-leg).

## Component View

```mermaid
flowchart TD
    A[Notebook / Script] --> B[Backtester<br/>backtesting/engine.py]
    B --> C[StrategyRunner Protocol<br/>backtesting/types.py]
    C -.implemented by.-> D[OptionsStrategyRunner<br/>strategies/options_core/strategy_runner.py]

    A --> E[VRPHarvestingSpec + make_vrp_strategy<br/>strategies/vrp_harvesting/strategy.py]
    E --> D

    D --> F[StrategySpec<br/>strategies/options_core/specs.py]
    D --> G[entry.py<br/>build EntryIntent]
    D --> H[sizing.py<br/>risk + margin sizing]
    D --> I[lifecycle.py<br/>open/mark/close]
    D --> J[runner.py<br/>single-position date loop]

    G --> K[selectors.py<br/>DTE/delta/liquidity selection]
    G --> L[types.py<br/>LegSpec/StructureSpec/EntryIntent]
    H --> M[options/*<br/>pricer/risk/margin models]
    I --> N[backtesting/margin.py<br/>margin lifecycle account]
```

## Runtime Sequence (One Backtest Run)

```mermaid
sequenceDiagram
    participant U as User Notebook
    participant B as Backtester
    participant S as OptionsStrategyRunner
    participant E as Entry Builder
    participant Z as Sizing
    participant L as Lifecycle

    U->>B: run(data, strategy, config)
    B->>S: run(SliceContext)
    S->>S: generate signals + apply filters
    loop each trading date
        alt position open
            S->>L: mark_position(...)
            L-->>S: mtm_record + optional trade close rows
        end
        alt entry date active and flat
            S->>E: build_entry_intent_from_structure(...)
            E-->>S: EntryIntent or None
            S->>Z: size_entry_intent_contracts(...)
            Z-->>S: contracts, risk/margin stats
            S->>L: open_position(...)
            L-->>S: entry mtm record
        end
    end
    S-->>B: trades, mtm
    B-->>U: trades, mtm
```

## Position State Machine

```mermaid
stateDiagram-v2
    [*] --> Flat
    Flat --> Open: signal on + valid entry intent + contracts > 0
    Open --> Open: daily mark-to-market
    Open --> Flat: Rebalance exit
    Open --> Flat: Max holding exit
    Open --> Flat: Margin call full liquidation
    Open --> Open: Margin call partial liquidation
    Flat --> Open: same-day reentry allowed by policy
    Flat --> Flat: same-day reentry not allowed
```

## Core Business Logic

### 1) Entry Construction (`entry.py`, `selectors.py`)

- Structure is defined as `StructureSpec(legs=tuple[LegSpec, ...])`.
- Legs can be grouped by `expiry_group` so related legs share one chosen expiry.
- For each group:
  - candidate expiries are filtered by DTE band,
  - each leg candidate is filtered by hard constraints (delta band, OI, volume, spread),
  - best quote per leg is scored,
  - best expiry is selected using DTE distance + weighted leg score.
- Fill policy is applied at structure level (`all_or_none` or `min_ratio`).

### 2) Sizing (`sizing.py`)

- Builds option-risk `OptionLeg` objects from selected legs.
- Computes:
  - risk-based contract limit (scenario worst loss),
  - margin-based contract limit (initial margin budget),
  - final contract count as constrained combination.

### 3) Lifecycle + Accounting (`lifecycle.py`, `backtesting/margin.py`)

- `open_position`: initializes Greeks, MTM baseline, margin account fields.
- `mark_position`: daily MTM, Greeks refresh, financing/margin updates.
- Applies exit rules (`exit_rules.py`) and emits trade rows on close.
- Supports forced partial/full liquidation from margin lifecycle.

### 4) Shared Loop (`runner.py`)

- Single-position date loop:
  - mark open position first,
  - optionally allow same-day reentry based on exit-type policy,
  - open new position only on active entry dates.

## Contracts and Boundaries

### Backtester Boundary

- `Backtester` depends on the minimal runtime contract:
  - `StrategyRunner` protocol in `backtesting/types.py`
  - required method: `run(ctx: SliceContext) -> (trades, mtm)`

This keeps `backtesting/` generic and independent from options-specific classes.

### Options Runtime Boundary

- `OptionsStrategyRunner` is one concrete implementation of `StrategyRunner`.
- It is configured entirely by `StrategySpec` and collaborators (pricer, risk,
  margin, entry/lifecycle policies).

## Strategy Preset Pattern

Current VRP preset:

- `VRPHarvestingSpec` defines business defaults
- `make_vrp_strategy(spec)` returns `OptionsStrategyRunner`

This pattern is reusable for future presets:

- Skew mispricing
- IV-RV mispricing
- Earnings vol-crush/rush structures

Each preset should primarily define:

1. signal + filters
2. structure spec (legs and constraints)
3. side resolver
4. risk/margin policy defaults

## Extension Points

Add new strategy behavior by configuration first:

- New structure: add `LegSpec` legs in `StructureSpec`
- New exit logic: implement `ExitRule` and attach to `ExitRuleSet`
- New sizing logic: provide `RiskSizer`/`RiskEstimator`/`MarginModel`
- New pricing model: pass another `PriceModel`

Only add new engine code when behavior cannot be expressed through these
contracts.

## Practical Rule of Thumb

- If logic is reusable across options strategies, it belongs in
  `strategies/options_core/`.
- If logic is a business preset or parameter bundle, it belongs in
  `strategies/<strategy_name>/strategy.py`.
