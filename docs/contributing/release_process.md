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

## Changelog Flow

- During normal PRs, user-visible changes should be added to `## [Unreleased]` in `CHANGELOG.md`.
- During release prep, move/copy relevant `Unreleased` bullets into `[0.x.y]`, set the release date, then reset `Unreleased`.

## Standard Release Workflow (PR-Protected `main`)

### 1) Create release branch from up-to-date `main`

```bash
git checkout main
git pull --ff-only
git checkout -b release/v0.x.y
```

### 2) Prepare release contents on the release branch

Required updates:

1. Bump version in `pyproject.toml`
2. Finalize `CHANGELOG.md`:
   - move/copy `Unreleased` entries into `[0.x.y]`
   - set release date
   - reset `Unreleased`
3. Update docs/examples if API/behavior changed

Ensure packaging tools are available in your environment:

```bash
pip install -e ".[dev,release]"
```

Run checks:

```bash
make check
make typecheck
make test-unit
make package-check
```

Then commit and push:

```bash
git add -A
git commit -m "chore(release): prepare v0.x.y"
git push -u origin release/v0.x.y
```

### 3) Open PR and merge into `main`

1. Open PR: `release/v0.x.y` -> `main`
2. Wait for required checks
3. Merge PR

No direct commit push to `main` for release prep.

### 4) Tag from local `main` after merge

```bash
git checkout main
git pull --ff-only
git tag -a v0.x.y -m "Release 0.x.y"
git push origin v0.x.y
```

Important:

- Create the tag from local `main` after pulling merged commits.
- Tag push triggers `.github/workflows/publish-testpypi.yml`.

### 5) Validate TestPyPI publish

1. Confirm `publish-testpypi.yml` succeeds.
2. Confirm smoke install step in that workflow succeeds.

Optional local install check:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple volatility-trading==0.x.y
```

### 6) Publish GitHub Release (triggers PyPI publish)

1. Create/publish GitHub Release for tag `v0.x.y`
2. This triggers `.github/workflows/publish-pypi.yml`
3. Confirm PyPI workflow succeeds

## Suggested Validation Commands

```bash
make check
make typecheck
make test-unit
make package-check
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

## Quick Checklist

1. `release/v0.x.y` branch from latest `main`
2. Bump version + changelog + release docs updates
3. PR to `main` and merge
4. `git pull --ff-only` on local `main`
5. Tag `v0.x.y` from local `main` and push tag
6. Validate TestPyPI workflow
7. Publish GitHub Release
8. Validate PyPI workflow

## Notes on Current Library State

- Options ETL/build pipeline is currently ORATS-first.
- Backtesting runtime is reusable, but options-chain inputs currently assume
  the project's existing options schema conventions.
- Public API boundaries for `0.1.x` are defined in
  [API Scope](../reference/api_scope.md).
- Research-oriented modules (for example `iv_surface`, `rv_forecasting`) are
  not yet treated as stable library API contracts.
- If schema boundaries change, document them in release notes and bump version
  according to the policy above.

## Failure and Recovery

If tag push succeeds, the tag exists on remote even when publishing workflows fail.
Workflow failure does not automatically remove or roll back a git tag.

### First classify the failure source

- **Code/package issue** (for example: failing tests, bad metadata, broken import, wrong versioning, packaging content bug).
- **Release infra/platform issue** (for example: Trusted Publishing not configured yet, temporary registry/network outage, rate limits, wrong GitHub Actions permission/config).

### A) Code/package issue

1. Fix on a normal fix branch (`fix/<topic>`), PR to `main`, merge.
2. Prepare a new release branch (`release/v0.x.(y+1)`) for:
   - version bump in `pyproject.toml`
   - `CHANGELOG.md` update
3. Merge release branch PR, then tag from updated local `main`.

Preferred outcome: publish with a new patch tag `v0.x.(y+1)`.

Minimal command flow:

```bash
git checkout main
git pull --ff-only
git checkout -b fix/<topic>
# apply code fix, then commit/push and merge PR

git checkout main
git pull --ff-only
git checkout -b release/v0.x.(y+1)
# bump pyproject version + update CHANGELOG, then commit/push and merge PR

git checkout main
git pull --ff-only
git tag -a v0.x.(y+1) -m "Release 0.x.(y+1)"
git push origin v0.x.(y+1)
```

### B) Release infra/platform issue (no package code change)

If package code is unchanged and failure is operational/config-related:

1. Fix release configuration/workflow issue (if needed) via PR to `main`.
2. Retry from GitHub Actions when safe:
   - rerun failed jobs for transient outages
   - rerun/retrigger release workflow after permission/setup fix
3. If retriggering on the same tag is not cleanly possible, cut a new patch tag.

### Reuse same tag vs bump patch version (policy)

- Enforced policy: release tags are immutable once pushed.
- Always bump patch version and create a new tag (`v0.x.(y+1)`).
- Do not delete/recreate release tags.
