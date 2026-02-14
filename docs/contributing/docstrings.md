# Docstring Guidelines

These guidelines define when and how to write docstrings across the codebase.

---

## TL;DR (Quick Rules)

1. Add module docstrings for entrypoints, public packages, and non-obvious modules.
2. Keep module docstrings short for simple files (1 to 3 lines). Skip only if truly redundant.
3. Use Google-style docstrings for boundary/public APIs and non-trivial logic.
4. For tiny obvious helpers, a one-liner (or no docstring) is acceptable.
5. Document side effects and assumptions when relevant (I/O, network, schema, units, timezone).
6. Update docstrings in the same PR when behavior changes.
7. Optimize for clarity, not verbosity.

---

## 1) Module-level docstrings

**Goal:** Answer “Why does this module exist?” for someone navigating the repo.

A good module docstring typically includes:
- **High-level responsibility** (what this module does)
- **Where it fits** in the system/pipeline (optional but useful for ETL/QC steps)
- **Key concepts** defined here (schemas, configs, QC thresholds, etc.)
- **Important invariants / assumptions** (schemas, units, time zones, idempotency)
- **Side effects** (what it reads/writes) when relevant

### How much to write

#### Public modules / public-facing code (preferred: full module docstring)
Write a fuller docstring when the module is:
- an entrypoint / pipeline step
- imported widely by other modules
- business-critical or non-obvious (QC rules, schema logic, vendor quirks)

Include, as applicable:
- inputs/outputs (`Reads:`, `Writes:`)
- run instructions (for entrypoints)
- invariants and error behavior (briefly)

#### Internal helper modules
For “small helper” modules with obvious intent:
- a one-liner is enough, or omit if it adds no value

---

## 2) Public vs. internal API

### Public API (inside this repo)
A function/class is treated as public if:
- it **does not** start with `_`, and
- it is intended to be imported and used outside its defining module
  (including re-export via `__init__.py`).

### What is mandatory vs optional

Mandatory full docstring (`Args`/`Returns`/`Raises`) for:
- CLI entrypoints and orchestration functions
- reusable library-style APIs used across packages
- functions/classes with side effects or non-obvious behavior

Recommended but can be lightweight for:
- simple pure helpers with obvious signatures and behavior
- private helpers (`_name`) unless logic is tricky or easy to misuse

---

## 3) Function and class docstrings (Google style)

Use Google-style docstrings for functions and classes.

```python
def load_options_chain(path: Path) -> pl.DataFrame:
    """Load and validate the options chain dataset.

    Args:
        path: Parquet file or directory containing the options chain.

    Returns:
        A Polars DataFrame with a normalized schema.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If required columns are missing.
    """
```

```python
@dataclass(frozen=True)
class EndpointSpec:
    """Endpoint contract used by download/extract runners.

    Attributes:
        path: Vendor endpoint path.
        strategy: Download strategy used by the runner.
        required: Required query fields for this endpoint.
    """
```

## Notes

- Keep docstrings concise; don’t restate obvious variable names.
- Include `Raises:` when exceptions are part of expected behavior for callers.
- Mention important assumptions (units, time zone, schema expectations) when relevant.
- `Returns:` can be omitted for `-> None` procedures when behavior is obvious.
- For trivial wrappers/helpers, a short one-liner is enough.

---

## 4) Module docstring templates

### Minimal template (most modules)

Use this when the module is straightforward and has no major side effects.

```python
"""Build the ORATS options chain dataset from extracted raw files.

Reads extracted ORATS API/FTP data and writes a normalized parquet dataset used by
downstream feature engineering and QC.
"""
```

## Entrypoint / pipeline step template (recommended for CLI modules)

Use this for modules that run work (download/extract/build/qc) and have side effects.

```python
"""ORATS API download step.

Reads:
- ORATS API using credentials from env (.env supported)
- YAML config specifying symbols, date range, and output paths

Writes:
- Raw payloads under <raw_dir> (see config)

Notes:
- Idempotent per (symbol, date); overwrites existing files if --force is set.
- Use --dry-run to validate config and credentials without downloading.
"""
```

## Complex business logic template (include assumptions)

Use this for QC rules, vendor quirks, schema normalization, and non-trivial math.

```python
"""QC checks for daily volatility features.

Validates schema, missingness, and numeric sanity bounds for ORATS-derived
features (orHv*, clsHv*, earnings-excluded variants). Produces a JSON report and
optional plots.

Assumptions:
- Input parquet contains one row per (ticker, tradeDate)
- tradeDate is YYYY-MM-DD in exchange local time
- Volatility fields are annualized decimals (e.g., 0.20 == 20%)
"""
```

## 5) Quality bar (what to avoid)

Avoid docstrings that:
- merely repeat the filename (“This module contains functions for…”)
- omit side effects for modules that read/write files or call external services
- hide important assumptions (units, schema invariants, time zones)

Prefer docstrings that:
- state purpose and placement in the pipeline
- make I/O and assumptions explicit when it matters
- help a new developer use the code correctly on the first try

---

## 6) Maintenance and linting

- Keep docstrings aligned with implementation in the same PR.
- If examples are included, keep them runnable.
- If enabling docstring linting, start small and strict only on high-value rules
  (for example: missing module/class/function docstrings on boundary APIs).
