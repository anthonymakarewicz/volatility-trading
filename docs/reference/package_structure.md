# Volatility Trading Package Structure

```plaintext
src/volatility_trading
├── __init__.py
├── apps
│   ├── __init__.py
│   ├── _cli.py
│   └── orats
│       ├── __init__.py
│       ├── build_daily_features.py
│       ├── build_options_chain.py
│       ├── download_api.py
│       ├── download_ftp.py
│       ├── extract_api.py
│       ├── extract_ftp.py
│       ├── qc_daily_features.py
│       └── qc_options_chain.py
├── backtesting
│   ├── __init__.py
│   ├── engine_prototype.py
│   ├── engine.py
│   ├── metrics.py
│   ├── plotting.py
│   └── types.py
├── cli
│   ├── __init__.py
│   ├── config.py
│   └── logging.py
├── config
│   ├── __init__.py
│   ├── constants.py
│   ├── instruments.py
│   ├── orats
│   │   ├── __init__.py
│   │   ├── api_schemas
│   │   │   ├── __init__.py
│   │   │   ├── dailies.py
│   │   │   ├── hvs.py
│   │   │   ├── monies_implied.py
│   │   │   ├── registry.py
│   │   │   └── summaries.py
│   │   ├── ftp_docs.py
│   │   ├── ftp_schemas.py
│   │   └── schema_spec.py
│   └── paths.py
├── datasets
│   ├── __init__.py
│   ├── daily_features.py
│   └── options_chain.py
├── etl
│   ├── __init__.py
│   ├── optionsdx_loader.py
│   └── orats
│       ├── __init__.py
│       ├── api
│       │   ├── __init__.py
│       │   ├── _client_helpers.py
│       │   ├── api.py
│       │   ├── client.py
│       │   ├── download
│       │   │   ├── __init__.py
│       │   │   ├── _handlers.py
│       │   │   ├── _helpers.py
│       │   │   └── run.py
│       │   ├── endpoints.py
│       │   ├── extract
│       │   │   ├── __init__.py
│       │   │   ├── _handlers.py
│       │   │   ├── _helpers.py
│       │   │   └── run.py
│       │   ├── io.py
│       │   └── types.py
│       ├── ftp
│       │   ├── __init__.py
│       │   ├── api.py
│       │   ├── download
│       │   │   ├── __init__.py
│       │   │   ├── _helpers.py
│       │   │   └── run.py
│       │   ├── extract
│       │   │   ├── __init__.py
│       │   │   ├── _helpers.py
│       │   │   └── run.py
│       │   └── types.py
│       ├── orats_io.py
│       ├── processed
│       │   ├── __init__.py
│       │   ├── daily_features
│       │   │   ├── __init__.py
│       │   │   ├── api.py
│       │   │   ├── config.py
│       │   │   ├── manifest.py
│       │   │   ├── steps
│       │   │   │   ├── __init__.py
│       │   │   │   ├── bounds.py
│       │   │   │   ├── canonicalize.py
│       │   │   │   ├── dedupe.py
│       │   │   │   ├── join.py
│       │   │   │   ├── output.py
│       │   │   │   └── scan.py
│       │   │   ├── transforms.py
│       │   │   └── types.py
│       │   ├── options_chain
│       │   │   ├── __init__.py
│       │   │   ├── api.py
│       │   │   ├── config.py
│       │   │   ├── io.py
│       │   │   ├── manifest.py
│       │   │   ├── steps
│       │   │   │   ├── __init__.py
│       │   │   │   ├── bounds.py
│       │   │   │   ├── dedupe.py
│       │   │   │   ├── enrich.py
│       │   │   │   ├── features.py
│       │   │   │   ├── filters.py
│       │   │   │   ├── greeks.py
│       │   │   │   ├── output.py
│       │   │   │   └── scan.py
│       │   │   ├── transforms.py
│       │   │   └── types.py
│       │   └── shared
│       │       ├── __init__.py
│       │       ├── bounds.py
│       │       ├── io.py
│       │       ├── log_fmt.py
│       │       ├── manifest.py
│       │       └── stats.py
│       └── qc
│           ├── __init__.py
│           ├── api.py
│           ├── common_helpers.py
│           ├── daily_features
│           │   ├── __init__.py
│           │   ├── hard
│           │   │   ├── __init__.py
│           │   │   └── specs.py
│           │   ├── info
│           │   │   ├── __init__.py
│           │   │   └── specs.py
│           │   ├── runner.py
│           │   ├── soft
│           │   │   ├── __init__.py
│           │   │   └── specs.py
│           │   └── specs_base.py
│           ├── hard
│           │   ├── __init__.py
│           │   ├── exprs.py
│           │   ├── spec_types.py
│           │   └── suite.py
│           ├── info
│           │   ├── __init__.py
│           │   ├── spec_types.py
│           │   ├── suite.py
│           │   └── summarizers.py
│           ├── options_chain
│           │   ├── __init__.py
│           │   ├── hard
│           │   │   ├── __init__.py
│           │   │   └── specs.py
│           │   ├── helpers.py
│           │   ├── info
│           │   │   ├── __init__.py
│           │   │   └── specs.py
│           │   ├── runner.py
│           │   └── soft
│           │       ├── __init__.py
│           │       └── specs.py
│           ├── plotting.py
│           ├── reporting.py
│           ├── runners.py
│           ├── serialization.py
│           ├── soft
│           │   ├── __init__.py
│           │   ├── dataset_checks
│           │   │   ├── __init__.py
│           │   │   ├── calendar_xnys.py
│           │   │   ├── rates.py
│           │   │   └── underlying_prices.py
│           │   ├── row_checks
│           │   │   ├── __init__.py
│           │   │   ├── arbitrage_bounds.py
│           │   │   ├── arbitrage_monotonicity.py
│           │   │   ├── arbitrage_parity.py
│           │   │   ├── expr_helpers.py
│           │   │   ├── greeks_iv.py
│           │   │   ├── quotes.py
│           │   │   └── volume_oi.py
│           │   ├── spec_types.py
│           │   ├── suite.py
│           │   ├── summarizers.py
│           │   └── utils.py
│           └── types.py
├── filters
│   ├── __init__.py
│   ├── base_filter.py
│   ├── fomc_filter.py
│   ├── ivp_filter.py
│   └── vix_filter.py
├── iv_surface
│   ├── __init__.py
│   ├── base_iv_surface_model.py
│   ├── base_xssvi.py
│   ├── essvi_model.py
│   ├── iv_surface_interpolator.py
│   ├── ssvi_model.py
│   ├── svi_model.py
│   └── term_structure.py
├── options
│   ├── __init__.py
│   ├── engines
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── binomial_tree_pricer.py
│   │   ├── bs_pricer.py
│   │   └── greek_approx_pricer.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── binomial_tree.py
│   │   └── black_scholes.py
│   └── types.py
├── rv_forecasting
│   ├── __init__.py
│   ├── data_loading.py
│   ├── features.py
│   ├── macro_features.py
│   ├── modelling
│   │   ├── __init__.py
│   │   ├── cross_validation.py
│   │   ├── data_processing.py
│   │   ├── evaluation.py
│   │   ├── feature_importance.py
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   ├── plotting.py
│   └── vol_estimators.py
├── signals
│   ├── __init__.py
│   ├── always_on_signal.py
│   ├── base_signal.py
│   └── z_score_signal.py
├── strategies
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── skew_mispricing
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── plotting.py
│   │   └── strategy.py
│   └── vrp_harvesting
│       ├── __init__.py
│       ├── features.py
│       ├── plotting.py
│       └── strategy.py
└── utils
    ├── __init__.py
    └── logging_config.py

47 directories, 196 files
```
