# Contributing

Thanks for contributing to this repository.

This project prioritizes correctness, reproducibility, and maintainability.
Use the docs below as the source of truth for day-to-day workflows:

- [Development workflow](docs/contributing/development.md)
- [Coding style](docs/contributing/coding_guide.md)
- [Testing standards](docs/contributing/testing_guide.md)
- [Test authoring](docs/contributing/test_authoring.md)
- [Docstrings](docs/contributing/docstrings.md)
- [Notebook workflow](docs/contributing/notebook_authoring.md)
- [Jupytext sync](docs/contributing/jupytext.md)
- [Release/versioning process](docs/contributing/release_process.md)

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

## Templates and Security Reporting

- Use the GitHub PR template as the default structure for PR summary, validation, and breaking-change notes.
- Use the GitHub issue templates when opening bug reports or feature requests so reports include the minimum context needed for triage.
- Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community behavior expectations.
- For suspected security issues, do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Changelog Policy

- If a PR is user-visible (feature, behavior change, fix, public API or workflow impact), update `CHANGELOG.md` under `## [Unreleased]` in that same PR.
- For tiny internal-only cleanup, changelog updates are optional.
- At release time, `Unreleased` entries are rolled into the versioned section (`[0.x.y]`) and `Unreleased` is reset.

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

## Commit Messages

Use Conventional Commits:

`<type>(scope): <summary>`

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`.

Use the narrowest stable domain scope that describes the changed surface.

Common scopes in this repo:

- `backtesting`
- `examples`
- `release`
- `orats`
- `etl`
- `qc`
- `cli`
- `notebooks`
- `docs`
- `ci`

Examples:

- `feat(backtesting): add pnl-per-contract stop-loss exits`
- `refactor(examples): align helpers with backtesting loaders`
- `docs(release): clarify pre-1.0 release timing`
- `feat(orats): add qc summary report generation`
- `fix(qc): handle empty bucket without crash`
- `docs(notebooks): clarify jupytext sync workflow`

Notes:

- Use imperative mood (`add`, `fix`, `remove`).
- Keep the summary <= 72 characters.
- If pre-commit modifies files during `git commit`, run `git add -A` and commit again.

For extended commit guidance and local quality gates, see
[Development Guide](docs/contributing/development.md).

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

- [Notebook Authoring Guide](docs/contributing/notebook_authoring.md)
- [Jupytext Workflow](docs/contributing/jupytext.md)

## Merge Strategy Policy

- Use `squash and merge` for feature/refactor PRs where one clean commit is preferred.
- Use `rebase and merge` when preserving each commit is useful and commit history is already clean.

## Workflow Trigger Matrix

- PR to `main` -> [.github/workflows/ci.yml](.github/workflows/ci.yml)
- Push to `main` -> [.github/workflows/ci.yml](.github/workflows/ci.yml), [.github/workflows/pages.yml](.github/workflows/pages.yml)
- Tag `v*.*.*` -> [.github/workflows/publish-testpypi.yml](.github/workflows/publish-testpypi.yml)
- Release published -> [.github/workflows/publish-pypi.yml](.github/workflows/publish-pypi.yml)

## Release Checklist

For releases, follow [Release Process](docs/contributing/release_process.md).

That page is the canonical maintainer workflow for:

- deciding when to cut `0.x.0` vs `0.x.y`
- preparing the release branch
- tagging and publish order
- TestPyPI/PyPI validation and recovery guidance

## Release Gate Policy

Release gating and publish-order rules are defined in
[Release Process](docs/contributing/release_process.md).

The key rule is:

- TestPyPI publish + smoke install must pass before GitHub Release/PyPI publish.
