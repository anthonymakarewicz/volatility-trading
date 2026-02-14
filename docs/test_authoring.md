# Test Authoring Guide

Use this guide when adding or refactoring tests under `tests/`.

For test layout and run commands, see [Testing Guide](testing_guide.md).

## Principles

- Keep tests deterministic, isolated, and fast.
- Prefer unit tests first; use integration tests only for cross-component paths.
- Assert behavior, not implementation details.
- Cover normal flow, edge cases, and failure paths.

## What to Test

For changed behavior, include tests for:

- happy path,
- boundary values,
- invalid inputs,
- empty/null inputs (if applicable),
- regression case for the bug being fixed.

## Fixtures and Reuse

- If setup repeats across 2+ tests in the same scope, move it to `conftest.py`.
- Keep fixtures small and composable.
- Use `tmp_path` for filesystem tests.
- Use `monkeypatch` / mocks for network, API, env vars, and external side effects.

## Parametrization

- Use `pytest.mark.parametrize` for similar input/output variants.
- Keep each parametrized case readable with explicit expected values.
- Split tests when scenarios have different intent.

## Determinism and Stability

- Set explicit seeds for randomized code paths.
- Avoid wall-clock dependencies unless specifically testing time behavior.
- Avoid real network calls in unit tests.
- Avoid reliance on machine-local state.

## Assertions and Quality

- Prefer exact assertions when stable.
- For floats, use tolerant checks (for example `pytest.approx`).
- Assert key side effects (written files, raised errors, logs) when relevant.
- Keep one clear behavioral intent per test function.

## Unit vs Integration

- Unit tests:
  - fast,
  - isolated from network/external services,
  - focused on one component/function.
- Integration tests:
  - cover CLI/pipeline wiring and cross-module behavior,
  - must be explicitly marked (`integration`),
  - can be slower, but still deterministic.

## Refactor Checklist

When refactoring tests:

1. remove duplicated setup into fixtures,
2. reduce brittle assertions,
3. keep names explicit (`test_<behavior>_<condition>`),
4. keep fixtures close to usage scope.
