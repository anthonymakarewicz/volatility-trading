# Coding Guide

General rules for writing and refactoring code in this repository (primarily
`src/volatility_trading/**`).

For docstring policy, see [Docstring Guidelines](docstrings.md).
For testing policy, see [Testing Guide](testing_guide.md) and
[Test Authoring Guide](test_authoring.md).

## Principles

- Optimize for clarity and correctness first, performance second.
- Keep diffs minimal and scoped to the change.
- Prefer explicitness over cleverness.
- Make decisions traceable (config, transforms, thresholds).

## Naming and Readability

- Use descriptive names (avoid single-letter variables outside math-heavy loops).
- Include units/horizons in names when relevant (`horizon_days`, `yte`, `dte`).
- Keep function responsibilities narrow; avoid “do everything” functions.
- Prefer small, pure helpers when it improves testability.

## Module Structure and Refactoring

- Keep modules cohesive: one clear responsibility per module.
- If a module grows large or mixes concerns, split it into submodules.
- Put shared types/config into `types.py` when they are imported across modules.
- Put shared, private helpers into `_helpers.py` within the local package.
- Put truly cross-cutting utilities into `src/volatility_trading/utils/`.

## Types and Interfaces

- Use type hints for public/boundary functions and non-trivial internal logic.
- Use `TypeAlias` for repeated complex types (mappings, typed dict-like payloads).
- Prefer importing library-provided types over re-declaring them locally.
- Avoid `cast(...)` unless you can justify why the runtime type is safe.

## Configuration, Constants, and “Types”

Use a consistent split:

- `constants.py`: immutable constants and small enums/literals.
- `paths.py`: filesystem path conventions and path roots.
- `types.py`: dataclasses and type aliases used across multiple modules.

Guidelines:

- If you pass 3+ related values through multiple functions, use a dataclass.
- Prefer `@dataclass(frozen=True)` for config-like objects.
- Keep defaults close to the boundary (CLI/YAML defaults) and log the resolved config.

## I/O, Side Effects, and Safety

- Use `pathlib.Path` for path operations.
- Prefer atomic writes when producing pipeline artifacts (write temp + rename).
- Make side effects explicit in docstrings and log key writes/paths.
- Avoid hidden global state; pass dependencies (paths, config, clients) explicitly.

## Errors and Logging

- Raise specific exceptions with actionable messages (include endpoint/ticker/path).
- Use `logging` (not `print`) in `src/` code.
- Log “shape” and key counts for ETL steps (rows read/written, failures).

## Data and Numerics

- Be explicit about time conventions and units (calendar vs trading days, annualization).
- When using float comparisons, use tolerances and explain why.
- Keep pandas/polars boundaries explicit and localized.

## Quality Gates

- Run the narrowest checks first (targeted tests), then broader checks if needed.
- Keep code formatted and lint-clean with the repo tooling (`ruff`, `pyright`).
