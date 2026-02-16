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
