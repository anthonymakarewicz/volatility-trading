# Contributing

Thanks for contributing to this repository.

This project prioritizes correctness, reproducibility, and maintainability.
Use the docs below as the source of truth for day-to-day workflows:

- Development workflow: `docs/contributing/development.md`
- Coding style: `docs/contributing/coding_guide.md`
- Testing standards: `docs/contributing/testing_guide.md`
- Test authoring: `docs/contributing/test_authoring.md`
- Docstrings: `docs/contributing/docstrings.md`
- Notebook workflow: `docs/contributing/notebook_authoring.md`
- Jupytext sync: `docs/contributing/jupytext.md`
- Release/versioning process: `docs/contributing/release_process.md`

## Branch and PR Workflow

Use short-lived feature branches and pull requests:

- Start from an up-to-date `main` before creating a branch:

```bash
git checkout main
git pull --ff-only
git checkout -b feature/<topic>
```

- Create a branch from `main`:
  - naming examples: `feature/<topic>`, `fix/<topic>`, `refactor/<topic>`
- Make changes, then commit and push the branch:

```bash
git add -A
git commit -m "type(scope): summary"
git push -u origin feature/<topic>
```

- Open a PR from the branch into `main`.
- Avoid direct pushes to `main` for non-trivial changes.

After a PR is merged, sync local `main` and clean up the merged branch:

```bash
git checkout main
git pull --ff-only
git branch -d feature/<topic>
git push origin --delete feature/<topic>
```

## PR Lifecycle

- After the first meaningful commit, open a Draft PR early to start CI and keep work visible.
- Each push to the PR branch triggers CI automatically on the PR (`pull_request` event).
- Before requesting review, update the PR title/description so scope and intent are explicit.
- If you need to switch context mid-work, either commit/push a small WIP commit on the branch or use `git stash` for temporary local changes.
- If WIP commits are noisy, clean history before merge (`rebase`) or use `squash and merge`.

## Branch Naming

Use branch names in the form:

`<type>/<topic>`

Recommended types:

- `feature`
- `fix`
- `refactor`
- `docs`
- `ci`
- `chore`
- `release`

Examples:

- `feature/optionsdx-adapter`
- `fix/ci-required-checks`
- `refactor/options-engine-types`
- `docs/release-workflow`
- `release/v0.1.0`

## Required Checks Before Merge

Before merging a PR, CI runs automatically on the PR branch and must pass:

- `CI / unit (pull_request)`
- `CI / integration (pull_request)`

Recommended local pre-checks before opening or updating a PR:

- Lint/format checks (`make lint`, `make format`, or `make check`)
- Type checks (`make typecheck`)
- Relevant tests at minimum (targeted unit tests), and broader test suite when changes are cross-cutting
- Any changed docs/examples should be updated in the same PR

For notebook edits, follow:

- `docs/contributing/notebook_authoring.md`
- `docs/contributing/jupytext.md`

## Commit Messages

Use Conventional Commits:

`<type>(scope): <summary>`

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`.

Preferred scopes: `orats`, `etl`, `qc`, `cli`, `notebooks`, `docs`, `ci`.

Examples:

- `feat(orats): add qc summary report generation`
- `fix(qc): handle empty bucket without crash`
- `docs(notebooks): clarify jupytext sync workflow`

Notes:

- Use imperative mood (`add`, `fix`, `remove`).
- Keep the summary <= 72 characters.
- If pre-commit modifies files during `git commit`, run `git add -A` and commit again.

For extended commit guidance and local quality gates, see
`docs/contributing/development.md`.

## Merge Strategy Policy

- Use `squash and merge` for feature/refactor PRs where one clean commit is preferred.
- Use `rebase and merge` when preserving each commit is useful and commit history is already clean.

## Workflow Trigger Matrix

- PR to `main` -> `.github/workflows/ci.yml`
- Push to `main` -> `.github/workflows/ci.yml`, `.github/workflows/pages.yml`
- Tag `v*.*.*` -> `.github/workflows/publish-testpypi.yml`
- Release published -> `.github/workflows/publish-pypi.yml`

## Release Checklist

For releases, follow `docs/contributing/release_process.md`.
Minimum checklist:

- Ensure working tree is clean and CI is green
- Update version (`0.x.y` while pre-1.0)
- Update `CHANGELOG.md`
- Create an annotated git tag (`v0.x.y`)
- Push the release tag (`git push origin v0.x.y`) to trigger `.github/workflows/publish-testpypi.yml`
- Validate TestPyPI publish and smoke-install workflow run
- Create/publish GitHub Release to trigger `.github/workflows/publish-pypi.yml`
- Validate PyPI publish workflow run

## Release Gate Policy

- TestPyPI publish + smoke install must pass before GitHub Release/PyPI publish.
- Release tag version must match `pyproject.toml` (`[project].version`).
- Package publishing is automated by GitHub Actions workflows, not manual `twine upload`.
