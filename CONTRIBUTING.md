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

- Create a branch from `main`:
  - naming examples: `feature/<topic>`, `fix/<topic>`, `refactor/<topic>`
- Open a PR from the branch into `main`.
- Prefer squash-merge so one PR maps to one clean history unit.
- Avoid direct pushes to `main` for non-trivial changes.

## Required Checks Before Merge

Before merging a PR, run and pass:

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

## Release Checklist

For releases, follow `docs/contributing/release_process.md`.
Minimum checklist:

- Ensure working tree is clean and CI is green
- Update version (`0.x.y` while pre-1.0)
- Update `CHANGELOG.md`
- Create an annotated git tag (`v0.x.y`)
- Create GitHub Release notes
- Publish to TestPyPI first, then PyPI after validation
