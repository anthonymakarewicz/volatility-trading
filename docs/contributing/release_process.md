# Release Process

This document defines the release and versioning workflow for this project.

## Scope

- Package: `volatility-trading`
- Current phase: pre-`1.0.0` (`0.x.y`)
- Source of package version: `setup.cfg` (`[metadata].version`)

## Versioning Policy (Pre-1.0)

Use `0.x.y` with these rules:

- `x` (minor): new features, significant refactors, and pre-1.0 breaking API changes
- `y` (patch): bug fixes, docs/test updates, and non-breaking internal cleanup

Guideline:

- If external usage or contracts may change, bump `x`.
- If behavior is compatible and scope is corrective, bump `y`.

## Branch and PR Expectations

Use a feature branch and PR for release-bound changes:

- `feature/*`, `fix/*`, or `refactor/*` branch from `main`
- PR into `main`
- Merge only when required checks pass

Avoid direct pushes to `main` for non-trivial changes.

## Release Checklist

1. Ensure CI is green and local quality gates pass.
2. Update docs/examples for any API or behavior changes.
3. Bump version in `setup.cfg`.
4. Update `CHANGELOG.md` (create it if missing).
5. Commit release preparation changes.
6. Create annotated tag `v0.x.y`.
7. Create GitHub Release with concise notes.
8. Publish to TestPyPI and validate install.
9. Publish to PyPI after TestPyPI validation.

## Suggested Validation Commands

```bash
make check
make typecheck
make test-unit
```

Optional broader gates:

```bash
make test
make ci
```

## Packaging and Publish Flow (Example)

Build artifacts:

```bash
python -m build
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

## Notes on Current Library State

- Options ETL/build pipeline is currently ORATS-first.
- Backtesting runtime is reusable, but options-chain inputs currently assume
  the project's existing options schema conventions.
- If schema boundaries change, document them in release notes and bump version
  according to the policy above.
