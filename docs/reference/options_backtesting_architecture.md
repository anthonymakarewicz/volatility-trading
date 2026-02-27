# Options Backtesting Architecture

This document describes how the current options backtesting stack is wired, from
`Backtester` orchestration to strategy-specific presets such as VRP harvesting.

## Scope

- Covered: options strategy runtime (`backtesting/` + `backtesting/options_engine/`)
- Not covered: data ETL/QC pipelines and model research notebooks

## Design Intent

The architecture separates:

1. **Backtest orchestration** (engine, execution loop, strategy-spec contract)
2. **Generic options execution runtime** (entry selection, sizing, lifecycle)
3. **Strategy presets** (VRP or future skew/IV-RV variants)

This keeps strategy-specific code minimal while reusing the same lifecycle and
accounting logic across structures (single-leg or multi-leg).

## Component View

```mermaid
flowchart TD
    A[User Entrypoint] --> B[Backtester<br/>backtesting/engine.py]
    A --> C[VRPHarvestingSpec + make_vrp_strategy<br/>strategies/vrp_harvesting/specs.py]
    C --> D[StrategySpec<br/>backtesting/options_engine/specs.py]
    B --> D
    B --> E[build_options_execution_plan<br/>backtesting/options_engine/strategy_runner.py]
    E --> F[entry.py<br/>build EntryIntent]
    E --> G[sizing.py<br/>risk + margin sizing]
    E --> H[lifecycle.py<br/>open/mark/close]
    B --> I[run_backtest_execution_plan<br/>engine-owned loop]

    F --> J[selectors.py<br/>DTE/delta/liquidity selection]
    F --> K[types.py<br/>LegSpec/StructureSpec/EntryIntent]
    G --> L[options/*<br/>pricer/risk/margin models]
    H --> M[backtesting/margin.py<br/>margin lifecycle account]
```

## Runtime Sequence (One Backtest Run)

```mermaid
sequenceDiagram
    participant U as User Entrypoint
    participant B as Backtester
    participant P as Plan Builder
    participant E as Entry Builder
    participant Z as Sizing
    participant L as Lifecycle

    U->>B: run(data, strategy, config)
    B->>P: build_options_execution_plan(...)
    P->>P: generate signals + apply filters
    loop each trading date
        alt position open
            B->>L: mark_position(...)
            L-->>B: mtm_record + optional trade close rows
        end
        alt entry date active and flat
            B->>E: build_entry_intent_from_structure(...)
            E-->>B: EntryIntent or None
            B->>Z: size_entry_intent(...)
            Z-->>B: contracts, risk/margin stats
            B->>L: open_position(...)
            L-->>B: entry mtm record
        end
    end
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

### 4) Runtime Loop (`engine.py`)

- Engine-owned single-position date loop (`engine.py`):
  - mark open position first,
  - optionally allow same-day reentry based on exit-type policy,
  - open new position only on active entry dates.

## Contracts and Boundaries

### Backtester Boundary

- `Backtester` consumes a concrete `StrategySpec`.
- `Backtester` compiles that spec into a `BacktestExecutionPlan` and owns the
  runtime execution loop.

### Options Runtime Boundary

- `backtesting/options_engine/` contains pure options-domain building blocks:
  entry selection, sizing, lifecycle, exit rules, and plan compilation.
- It does not own top-level run orchestration.

## Strategy Preset Pattern

Current VRP preset:

- `VRPHarvestingSpec` defines business defaults
- `make_vrp_strategy(spec)` returns a concrete `StrategySpec`

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
  `backtesting/options_engine/`.
- If logic is a business preset or parameter bundle, it belongs in
  `strategies/<strategy_name>/specs.py`.
