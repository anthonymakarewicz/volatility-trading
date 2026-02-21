# Daily Features Column Onboarding

This guide explains how to add columns to the processed ORATS
`daily_features` dataset in two common situations:

1. The column already exists in intermediate parquet and you now want it in
   processed daily features.
2. The column comes from a new API endpoint that is not yet modeled in
   `api_schemas` (and therefore not extracted to intermediate yet).

Use this doc as the operational checklist to keep schema, processing, QC, and
tests aligned.

## Decide First: Should It Be in `daily_features`?

Add it to processed `daily_features` when:

- the signal is daily-frequency and broadly useful for downstream models;
- multiple modules/strategies will reuse it;
- you want QC and a stable schema contract around it.

Keep it only in intermediate when:

- it is endpoint-specific scratch data;
- it is experimental and not ready for shared use;
- you need a short-lived analysis-only field.

## Naming Convention (Canonical)

Follow existing ORATS naming style:

- prices: `*_price` (for example `adjusted_close_price`);
- volumes: `*_volume`;
- dates/timestamps: `*_date`, `*_ts`;
- use `adjusted_*` vs `unadjusted_*` when both variants exist.

## Path A: Promote Existing Intermediate Columns

Use this path when the endpoint is already extracted and canonical columns
already exist in intermediate parquet.

1. Confirm the canonical column names in intermediate output.
   - Check endpoint parquet under
     `data/intermediate/orats/api/endpoint=<endpoint>/underlying=<TICKER>/part-0000.parquet`.
2. Add columns to processed daily-features config.
   - Update `src/volatility_trading/etl/orats/processed/daily_features/config.py`.
   - Add to `DAILY_FEATURES_ENDPOINT_COLUMNS["<endpoint>"]`.
   - Add to `DAILY_FEATURES_CORE_COLUMNS` if they are part of default output.
3. If needed, update default endpoint list for daily-features build.
   - `src/volatility_trading/etl/orats/processed/daily_features/api.py`
   - `src/volatility_trading/apps/orats/build_daily_features.py`
   - `config/orats/daily_features_build.yml`
4. Add/update QC coverage.
   - Start with non-negativity checks in
     `src/volatility_trading/etl/orats/qc/daily_features/hard/specs.py`.
   - Update column groupings in
     `src/volatility_trading/etl/orats/qc/daily_features/specs_base.py`.
5. Add tests for config/schema contracts and QC grouping behavior.

## Path B: Onboard a New API Endpoint

Use this path when the endpoint exists in ORATS but is not yet implemented in
`api_schemas`.

1. Ensure endpoint contract exists in
   `src/volatility_trading/etl/orats/api/endpoints.py`.
2. Add a schema module in
   `src/volatility_trading/config/orats/api_schemas/`.
   - Define vendor dtypes.
   - Define date/datetime vendor columns.
   - Add vendor-to-canonical rename mapping.
   - Define keep columns (for initial rollout, keeping all canonical fields is OK).
   - Add `bounds_null_canonical` and/or `bounds_drop_canonical`.
3. Register the schema in
   `src/volatility_trading/config/orats/api_schemas/registry.py`.
4. Run extract for the endpoint to produce intermediate parquet.
5. Follow Path A to promote selected columns into processed daily features.
6. Add QC checks and tests as above.

## Bounds Guidance

For daily-feature ingestion:

- use `bounds_null_canonical` for plausibility checks that should keep rows but
  null bad values (recommended default);
- use `bounds_drop_canonical` only for structural invalidity where a full row
  is not trustworthy.

Examples:

- `*_price`: `[0, very_large_cap]`;
- `*_volume`: `[0, very_large_cap]`;
- vol/rate fields should match existing project conventions.

## QC Guidance

Minimum recommended QC when adding new daily-feature columns:

1. HARD checks:
   - key columns not null (already in place);
   - non-negative checks for price/volume style columns.
2. INFO checks:
   - keep new columns in core numeric stats output.
3. SOFT checks:
   - add only when you have clear domain thresholds (optional for first pass).

## Definition of Done

A daily-features column onboarding is complete when:

1. API schema is defined/registered (if new endpoint).
2. Intermediate extract contains canonical columns.
3. Processed daily-features build includes new columns.
4. QC checks cover the columns.
5. Unit tests cover new schema/config/QC expectations.
6. Docs are updated:
   - `docs/reference/configs.md`
   - `docs/reference/data_pipeline.md` (if operational flow changed)
   - `docs/reference/package_structure.md` (if new module added)

## Suggested Verification Commands

```bash
orats-api-extract --config config/orats/api_extract.yml --endpoint <endpoint> --dry-run
orats-build-daily-features --config config/orats/daily_features_build.yml --dry-run
orats-qc-daily-features --config config/orats/qc_daily_features.yml --dry-run

pytest -q tests/unit/etl/orats/api
pytest -q tests/unit/etl/orats/processed
pytest -q tests/unit/etl/orats/qc
```
