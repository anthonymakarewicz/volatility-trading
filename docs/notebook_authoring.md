# Notebook Design Instructions for LLMs

Use this guide to design or refactor any research notebook in this repository.
This is a general style spec, not tied to one notebook.

## Goal

Write notebooks that are:

- educational (clear intuition),
- operational (explicit methods and decisions),
- reproducible (easy to rerun and audit).

## Core Principles

- Keep markdown and code balanced (`~45%-55%` markdown is a good target).
- Use numbered, logical sections (`1`, `2`, `3`, ...).
- Pair every important output with interpretation.
- Keep explanations concise and specific.
- Prefer clear evidence over long prose.

## Section Structure

Use this sequence unless scope is intentionally smaller:

1. `Motivation and questions`
2. `Data and setup`
3. `Definitions / methods`
4. `Diagnostics / stylized facts`
5. `Modeling or strategy design`
6. `Evaluation / robustness`
7. `Conclusion`

## Cell Rhythm

For each analytical block:

1. context markdown,
2. code/plot/table cell,
3. takeaway markdown.

Each takeaway should answer at least one:

- What did we observe?
- Why does it matter?
- What decision follows?

## Equation Policy

Use LaTeX only when it adds precision:

- targets,
- estimators,
- model equations,
- evaluation metrics.

Rules:

- keep equations selective (not overloaded),
- define symbols once,
- keep formulas aligned with implemented code.

## Writing and Interpretation Rules

- One idea per paragraph.
- Use bullets for assumptions, decisions, and conclusions.
- Avoid generic comments like "the plot looks good".
- Prefer decision-oriented statements:
  - "This supports using ..."
  - "This suggests dropping ..."
  - "This is regime-dependent, so ..."

## Decision Traceability

Document non-trivial choices explicitly:

- feature inclusion/exclusion,
- transformations,
- threshold choices,
- model selection criteria.

Recommended marker:

- `#### Transformation Decisions`

## Accuracy Guardrails

Before writing technical claims:

1. verify against code outputs,
2. verify variable names and definitions,
3. avoid claims stronger than shown evidence.

Never invent formulas, parameter values, feature names, or results.

## Reproducibility Rules

- Keep random seeds explicit when randomness is used.
- Keep horizon/window definitions explicit and consistent.
- Avoid hidden state assumptions.
- Prefer reusable project functions over duplicated notebook logic.

## Jupytext Rules

After notebook edits:

1. sync pair (`jupytext --sync notebooks/<name>.ipynb`),
2. keep `.ipynb` and `.py` consistent and committed together.

## Prompt Template for New Chat

```text
You are editing a research notebook in a Jupytext-paired repository.

Follow these rules:
- Keep markdown/code balanced and section flow logical.
- For each major block: context -> code output -> takeaway.
- Use LaTeX only for core formulas (targets/methods/metrics).
- Keep explanations concise and decision-oriented.
- Document transformation/selection decisions explicitly.
- Do not invent formulas, parameters, feature names, or results.
- Ensure claims match actual code outputs.
- Preserve reproducibility and leakage-safe methodology.

Output:
- clear section structure,
- selective equations,
- concise interpretation after key plots/tables.
```
