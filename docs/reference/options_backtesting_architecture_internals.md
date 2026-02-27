# Options Backtesting Architecture Internals

This document is the developer-facing deep dive for the options backtesting
stack. It explains how modules interact, which dataclasses cross boundaries,
and why the current architecture is organized this way.

For a shorter overview, see
[`docs/reference/options_backtesting_architecture_overview.md`](options_backtesting_architecture_overview.md).

## Scope

- Covered:
  - `src/volatility_trading/backtesting/engine.py`
  - `src/volatility_trading/backtesting/options_engine/**`
  - `src/volatility_trading/backtesting/margin.py`
  - key domain types under `src/volatility_trading/options/types.py`
- Not covered:
  - ETL/QC pipelines
  - forecasting/research notebooks

## 1) Layered Component Map

```mermaid
flowchart LR
    subgraph UserLayer[User / Strategy Presets]
        U1[Notebook / Script / CLI]
        U2[VRPHarvestingSpec\nstrategies/vrp_harvesting/specs.py]
        U3[StrategySpec\noptions_engine/specs.py]
        U4[BacktestRunConfig\nbacktesting/config.py]
    end

    subgraph Orchestrator[Backtesting Orchestrator]
        O1[Backtester\nbacktesting/engine.py]
        O2[run_backtest_execution_plan\nbacktesting/engine.py]
    end

    subgraph PlanCompiler[Options Plan Compiler]
        P1[build_options_execution_plan\noptions_engine/plan_builder.py]
        P2[SinglePositionExecutionPlan\noptions_engine/contracts.py]
        P3[SinglePositionHooks\noptions_engine/contracts.py]
    end

    subgraph RuntimeDomain[Options Runtime Domain]
        R1[entry.py\nnormalize_signals_to_on\nbuild_entry_intent_from_structure]
        R2[sizing.py\nsize_entry_intent]
        R3[lifecycle.py\nPositionLifecycleEngine\nopen_position/mark_position]
        R4[exit_rules.py\nExitRuleSet / SameDayReentryPolicy]
        R5[selectors.py\nquote/expiry selection]
        R6[economics.py\neffective side/units\ncommission helpers]
        R7[outputs.py\nbuild_options_backtest_outputs]
    end

    subgraph LifecycleInternals[_lifecycle Internal Modules]
        L1[valuation.py\nresolve_mark_valuation\nupdate_position_mark_state]
        L2[margining.py\nevaluate_entry_margin\nevaluate_mark_margin]
        L3[record_builders.py\nbuild_entry_record\nbuild_mark_record\nbuild_trade_record]
        L4[runtime_state.py\nEntry/Mark snapshots]
        L5[opening.py / marking.py / transitions.py\nstate/context/transition handlers]
    end

    subgraph RuntimeContracts[Public Runtime Contracts]
        C1[state.py\nOpenPosition\nPositionEntrySetup\nLifecycleStepResult]
        C2[records.py\nMtmRecord\nTradeRecord]
    end

    subgraph ExternalDeps[Cross-Package Dependencies]
        X1[backtesting/margin.py\nMarginAccount / MarginPolicy]
        X2[options/*\nPriceModel / MarginModel / Risk stack]
        X3[options/types.py\nMarketState / Greeks]
    end

    U1 --> U2 --> U3
    U1 --> U4
    U1 --> O1
    O1 --> P1
    O1 --> O2
    U3 --> P1
    U4 --> P1
    P1 --> P2
    P1 --> P3
    P1 --> R1
    P1 --> R2
    P1 --> R3
    P1 --> R4
    P1 --> R7
    R1 --> R5
    R2 --> R6
    R3 --> R6
    R3 --> L1
    R3 --> L2
    R3 --> L3
    R3 --> L4
    R3 --> C1
    R3 --> C2
    L2 --> X1
    R2 --> X2
    R3 --> X2
    L1 --> X3
    C2 --> X3
```

## 2) Core Runtime Contracts (Typed Boundaries)

The engine loop does not call entry/sizing/lifecycle directly. It runs a typed
plan with typed hooks.

```mermaid
flowchart TD
    A[StrategySpec] --> B[build_options_execution_plan]
    A2[BacktestRunConfig] --> B
    B --> C[SinglePositionExecutionPlan]
    C --> D[SinglePositionHooks]

    D --> D1["mark_open_position<br/>(open_position, date, equity) -> lifecycle_step_result"]
    D --> D2["prepare_entry<br/>(date, equity) -> entry_setup or None"]
    D --> D3["open_position<br/>(entry_setup, equity) -> (open_position, mtm_record)"]
    D --> D4["can_reenter_same_day<br/>(trade_rows) -> bool"]

    C --> E["build_outputs<br/>(trade_records, mtm_records, initial_capital) -> (trades_df, mtm_df)"]
```

### Strategy vs Run-Config Dataclasses

```mermaid
flowchart LR
    S[StrategySpec] --> S1[LifecycleConfig]
    S --> S2[SizingPolicyConfig]
    R[BacktestRunConfig] --> R1[AccountConfig]
    R --> R2[ExecutionConfig]
    R --> R3[BrokerConfig]
    R --> R4[ModelingConfig]
    R3 --> R31[MarginConfig]
```

### Dataclass Composition Map

```mermaid
flowchart LR
    S1[PositionEntrySetup] --> S2[EntryIntent]
    S3[OpenPosition] --> S2
    S4[LifecycleStepResult] --> S3
    S4 --> S5[MtmRecord]
    S4 --> S6[TradeRecord]

    S5 --> S7[MarketState]
    S5 --> S8[Greeks]
    S5 --> S9[MtmMargin]
    S9 --> S10[MarginCore]
```

## 3) End-to-End Runtime Sequence

```mermaid
sequenceDiagram
    participant U as User Entrypoint
    participant B as Backtester
    participant PB as build_options_execution_plan
    participant E as Entry + Selectors
    participant Z as Sizing
    participant L as PositionLifecycleEngine
    participant K as Engine Loop (run_backtest_execution_plan)

    U->>B: Backtester.run()
    B->>PB: build_options_execution_plan(spec, data, cfg, capital)
    PB->>PB: signal.generate_signals(...)
    PB->>PB: apply filters
    PB->>PB: compute trading_dates + active_signal_dates
    PB->>L: instantiate lifecycle engine
    PB-->>B: SinglePositionExecutionPlan

    B->>K: run_backtest_execution_plan(plan)
    loop each trading date
        alt open_position exists
            K->>L: mark_position(position, date, options, cfg, equity)
            L-->>K: LifecycleStepResult(position|None, mtm_record, trade_rows)
            K->>K: append MTM/trades, update equity
            alt position still open
                K->>K: continue next date
            else closed
                K->>K: evaluate can_reenter_same_day(trade_rows)
            end
        end

        alt active entry date and flat
            K->>E: build_entry_intent_from_structure(...)
            E-->>K: EntryIntent or None
            K->>Z: size_entry_intent(...)
            Z-->>K: PositionEntrySetup or None
            K->>L: open_position(setup, cfg, equity)
            L-->>K: (OpenPosition, entry MtmRecord)
            K->>K: append entry MTM, update equity
        end
    end
    K-->>B: build_outputs(trade_records, mtm_records, initial_capital)
    B-->>U: (trades_df, mtm_df)
```

## 4) `mark_position` Decision Flow (Lifecycle Internals)

```mermaid
flowchart TD
    A[mark_position] --> B[resolve_mark_valuation]
    B --> C[maybe_refresh_margin_per_contract]
    C --> D[evaluate_mark_margin]
    D --> E[build_mark_record]
    E --> F{Forced liquidation?}

    F -- Yes --> G[transition_forced_liquidation]
    G --> H[return LifecycleStepResult]

    F -- No --> I{Missing quote?}
    I -- Yes --> J[update_position_mark_state]
    J --> K[return open position + no trade row]

    I -- No --> L[exit_rule_set.evaluate]
    L --> M{Exit triggered?}
    M -- No --> N[update_position_mark_state]
    N --> O[return open position + no trade row]

    M -- Yes --> P[transition_standard_exit]
    P --> Q[build_trade_record + close MTM fields]
    Q --> R[return closed position + trade row]
```

## 5) Module Responsibilities and Ownership

| Module | Owns | Does Not Own |
|---|---|---|
| `backtesting/engine.py` | orchestration loop, equity progression, top-level `Backtester.run()` | option quote selection, risk/margin formulas, trade construction details |
| `options_engine/contracts.py` | typed kernel boundary between engine and options runtime | business logic |
| `options_engine/plan_builder.py` | compile `StrategySpec` into executable plan | strategy execution loop and tabular aggregation |
| `options_engine/outputs.py` | convert typed records into canonical `trades`/`mtm` DataFrames | signal generation, entry/mark/exit lifecycle logic |
| `options_engine/entry.py` + `selectors.py` | build `EntryIntent` from chain + structure constraints | account/margin lifecycle |
| `options_engine/economics.py` | shared leg-side/units and per-structure commission helpers | signal/entry/exit orchestration |
| `options_engine/sizing.py` | contract count decision from risk/margin constraints | position mark/exit loop |
| `options_engine/lifecycle.py` | open/mark/close logic, forced liquidation, exit handling | signal generation, global orchestration |
| `options_engine/_lifecycle/*` | valuation/margin/record/state internals | plan compilation and outer loop |

## 6) Why This Architecture (Design Choices)

### Choice A: Engine owns execution loop

- Single place for date iteration and equity progression.
- Strategy modules cannot diverge in loop semantics.
- Easier to evolve to multi-position at one boundary.

### Choice B: Plan builder compiles plan instead of executing

- Clean separation between “what to run” and “how to run”.
- Allows unit testing plan compilation independent of runtime loop.
- Keeps strategy presets thin and declarative.

### Choice C: Concrete options-typed contracts (not broad `Any`)

- Stronger Pyright guarantees across engine <-> lifecycle boundary.
- Better IDE assistance for hooks and outputs.
- Fewer silent runtime shape mismatches.

### Choice D: Lifecycle internals split into `_lifecycle/*`

- Keeps `lifecycle.py` focused on control flow and business decisions.
- Valuation, margining, and serialization evolve independently.
- Tests can target specific responsibilities.

## 7) Invariants (Current Behavior)

1. One open position max at a time (single-position kernel).
2. MTM rows are appended daily for open positions and entry dates.
3. Equity progression is driven by cumulative `delta_pnl`.
4. Same-day reentry is governed exclusively by `reentry_policy`.
5. Final outputs are normalized pandas DataFrames (`trades`, `mtm`).

## 8) Extension Playbook

### Add a new options strategy preset

- Add preset spec/factory under `strategies/<name>/specs.py`.
- Output must be a valid `StrategySpec`.
- Reuse existing entry/sizing/lifecycle modules first.

### Add new exit behavior

- Implement new `ExitRule` in `exit_rules.py`.
- Register via `ExitRuleSet` in the strategy spec.

### Add new sizing behavior

- Update strategy policy in `StrategySpec.sizing` and/or provide another `RiskBudgetSizer`.
- Keep contract count decision in `size_entry_intent`.

### Add new pricing/risk engines

- Configure `BacktestRunConfig.modeling` (`pricer`, `scenario_generator`, `risk_estimator`).
- Keep strategy presets focused on business policy, not environment wiring.

### Add new margin rules

- Configure `BacktestRunConfig.broker.margin` (`model`, `policy`).
- Keep margin lifecycle behavior in `backtesting/margin.py` and lifecycle integration code.

### Add new accounting fields

- Add to public record dataclasses (`MtmRecord`/`TradeRecord`) first.
- Update serializer in `_lifecycle/record_builders.py`.
- Keep loop contract unchanged unless boundary shape truly changes.

### Prepare for multi-position support (future)

- Replace single `open_position` in engine loop with a position book.
- Generalize hooks to mark/open/close across multiple positions.
- Preserve existing plan-compiler boundary where possible.

## 9) File-Level Reading Order (Recommended)

1. `backtesting/engine.py`
2. `options_engine/contracts.py`
3. `options_engine/plan_builder.py`
4. `options_engine/outputs.py`
5. `options_engine/specs.py`
6. `options_engine/entry.py`
7. `options_engine/sizing.py`
8. `options_engine/lifecycle.py`
9. `options_engine/state.py`
10. `options_engine/records.py`
11. `options_engine/_lifecycle/runtime_state.py`
12. `options_engine/_lifecycle/valuation.py`
13. `options_engine/_lifecycle/margining.py`
14. `options_engine/_lifecycle/record_builders.py`
15. `options_engine/_lifecycle/opening.py`
16. `options_engine/_lifecycle/marking.py`
17. `options_engine/_lifecycle/transitions.py`
