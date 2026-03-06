# Docstring Guidelines

Use this guide for all Python modules in the repo.

For broader code style, see [Coding Guide](coding_guide.md).

## Quick Rules

1. Document module purpose when a file is public, non-obvious, or has side effects.
2. Use Google-style docstrings for public/boundary APIs and non-trivial logic.
3. Keep tiny obvious private helpers minimal (one-liner or no docstring).
4. Document assumptions that matter (units, timezone, schema shape, idempotency).
5. Keep docstrings updated in the same PR as behavior changes.

## What Needs a Full Docstring

Use full `Args`/`Returns`/`Raises` docstrings for:
- public functions/classes used across modules
- entrypoints and orchestration code
- functions with side effects (I/O, network, writes)
- logic that is easy to misuse or interpret incorrectly

Lightweight docstrings are fine for:
- private helpers with obvious behavior
- simple wrappers where signature and name are self-explanatory

## Module Docstrings

Module docstrings should answer: "Why does this module exist?"

Prefer 1 to 5 lines for most modules. Include extra detail only when needed:
- role in the architecture/pipeline
- important invariants or assumptions
- key side effects (`Reads`, `Writes`) for execution modules

Minimal example:

```python
"""Build execution plans for options backtesting.

Transforms strategy specs and run config into typed hooks used by the engine loop.
"""
```

Entrypoint example:

```python
"""CLI entrypoint for preparing OptionsDX panel data.

Reads raw files from configured paths and writes normalized parquet outputs.
"""
```

## Function/Class Format (Google Style)

```python
def load_options_chain(path: Path) -> pd.DataFrame:
    """Load and validate an options chain dataset.

    Args:
        path: File or directory path.

    Returns:
        DataFrame with canonical columns and normalized dtypes.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
```

Notes:
- Include `Raises` when callers should handle specific failure modes.
- `Returns` may be omitted for `-> None` when behavior is obvious.
- For dataclasses, use an `Attributes` section only when field meaning is not obvious.

## Quality Bar

Avoid:
- repeating the function name without adding meaning
- documenting obvious implementation details line by line
- hiding critical assumptions (units/time conventions/schema)

Prefer:
- concise intent + caller-relevant behavior
- explicit side effects for boundary code
- terminology consistent with type hints and config names

## Maintenance

- Treat docstrings as part of the contract.
- Update them whenever behavior, assumptions, or side effects change.
- Keep examples runnable when included.
