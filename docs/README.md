# Documentation Index

Use this page as the canonical entrypoint for repository documentation.

## Getting Started

- [Development Guide](contributing/development.md): local environment, core workflow, and CI-aligned checks.
- [Release Process](contributing/release_process.md): versioning policy (`0.x.y`), tags, and publish flow.
- [API Scope](reference/api_scope.md): explicit public vs internal API boundaries for `0.5.x`.
- [Examples](../examples/README.md): runnable end-to-end and focused backtesting examples.
- [Troubleshooting](troubleshooting.md): routing page to scoped troubleshooting guides.

## Engineering Standards

- [Coding Guide](contributing/coding_guide.md): code authoring and refactoring rules.
- [Docstring Guidelines](contributing/docstrings.md): module/function/class docstring policy.
- [Daily Features Onboarding](contributing/daily_features_onboarding.md): checklist for adding new daily-features columns/endpoints.

## Data and Pipeline

- [Data Pipeline](reference/data_pipeline.md): end-to-end ORATS runbook and data flow.
- [Backtesting Architecture Overview](reference/backtesting/architecture_overview.md): high-level runtime architecture, typed boundaries, and extension points.
- [Backtesting Architecture Internals](reference/backtesting/architecture_internals.md): deeper developer-facing module and contract map for the options runtime.
- [Backtest Runner Workflows](reference/backtesting/workflows.md): YAML workflow schema for `backtest-run` and the shipped config templates.
- [Options Engine Hedging](reference/backtesting/hedging.md): fixed-band and WW-style delta hedging configuration and runtime semantics.
- [Options Engine Option Execution](reference/backtesting/option_execution.md): option fill models, transaction-cost attribution, and model injection boundary.
- [Options Data Adapters](reference/options_data_adapters.md): canonical
  options-chain contract and source adapter usage.
- [ORATS Troubleshooting](reference/orats_troubleshooting.md): ORATS pipeline/app-specific failures and fixes.
- [Entrypoints](reference/entrypoints.md): CLI commands and shared flags.
- [Configs](reference/configs.md): YAML schema reference and path keys.
- [Package Structure](reference/package_structure.md): package/module layout.

## Testing

- [Testing Guide](contributing/testing_guide.md): test layout, commands, and CI behavior.
- [Test Authoring Guide](contributing/test_authoring.md): quality rules for writing and refactoring tests.

## Notebooks and Reports

- [Notebook Catalog](research/notebooks.md): notebook inventory and published report links.
- [Research Results](research/results.md): detailed experiment outcomes and strategy result summaries.
- [Notebook Authoring Guide](contributing/notebook_authoring.md): notebook design standards.
- [Jupytext Workflow](contributing/jupytext.md): pairing and sync workflow.
- [Publishing](research/publishing.md): GitHub Pages export and deployment flow.
