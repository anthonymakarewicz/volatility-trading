# Release Process

This document defines the release and versioning workflow for this project.

## Scope

- Package: `volatility-trading`
- Current phase: pre-`1.0.0` (`0.x.y`)
- Source of package version: `pyproject.toml` (`[project].version`)

## Versioning Policy (Pre-1.0)

Use `0.x.y` with these rules:

- `x` (minor): new features, significant refactors, and pre-1.0 breaking API changes
- `y` (patch): bug fixes, docs/test updates, and non-breaking internal cleanup

Guideline:

- If external usage or contracts may change, bump `x`.
- If behavior is compatible and scope is corrective, bump `y`.

## Release Checklist

1. Ensure CI is green and local quality gates pass.
2. Update docs/examples for any API or behavior changes.
3. Bump version in `pyproject.toml`.
4. Update `CHANGELOG.md` (create it if missing).
5. Commit release preparation changes.
6. Create annotated tag `v0.x.y`.
7. Push tag `v0.x.y` to trigger `.github/workflows/publish-testpypi.yml`.
8. Validate TestPyPI workflow publish + smoke install.
9. Create/publish GitHub Release with concise notes to trigger `.github/workflows/publish-pypi.yml`.
10. Validate PyPI publish workflow run.

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

## Packaging and Publish Flow

Publishing is automated via GitHub Actions:

- Tag `v*.*.*` push -> `.github/workflows/publish-testpypi.yml`
- `release: published` event -> `.github/workflows/publish-pypi.yml`

Both workflows build artifacts and run `twine check` before publishing.
No manual `twine upload` is required for the standard release path.

## Notes on Current Library State

- Options ETL/build pipeline is currently ORATS-first.
- Backtesting runtime is reusable, but options-chain inputs currently assume
  the project's existing options schema conventions.
- If schema boundaries change, document them in release notes and bump version
  according to the policy above.
