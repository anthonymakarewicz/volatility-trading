# Volatility Trading Project Structure

```plaintext
.
├── __init__.py
├── backtesting
│   ├── __init__.py
│   ├── engine_prototype.py
│   ├── engine.py
│   ├── metrics.py
│   ├── plotting.py
│   └── types.py
├── config
│   ├── __init__.py
│   ├── constants.py
│   ├── instruments.py
│   ├── orats
│   │   ├── __init__.py
│   │   ├── api_schemas.py
│   │   ├── ftp_docs.py
│   │   ├── ftp_schemas.py
│   │   └── schema_spec.py
│   ├── orats_processed_features.py
│   └── paths.py
├── datasets
│   ├── __init__.py
│   ├── daily_features.py
│   └── options_chain.py
├── etl
│   ├── __init__.py
│   ├── common
│   │   ├── io.py
│   │   ├── logging.py
│   │   └── manifest.py
│   ├── optionsdx_loader.py
│   └── orats
│       ├── __init__.py
│       ├── api
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── downloader.py
│       │   ├── endpoints.py
│       │   ├── extractor.py
│       │   └── io.py
│       ├── ftp
│       │   ├── __init__.py
│       │   ├── downloader.py
│       │   └── extractor.py
│       ├── io
│       │   └── atomic_write.py
│       ├── orats_io.py
│       ├── processed
│       │   ├── __init__.py
│       │   ├── daily_features_builder.py
│       │   ├── options_chain
│       │   │   ├── __init__.py
│       │   │   ├── builder.py
│       │   │   ├── config.py
│       │   │   ├── io.py
│       │   │   ├── steps.py
│       │   │   ├── transforms.py
│       │   │   └── types.py
│       │   └── options_chain_builder.py
│       └── qc
│           ├── __init__.py
│           ├── checks_hard.py
│           ├── checks_info.py
│           ├── checks_soft.py
│           ├── options_chain.py
│           ├── orats_qc_plotting.py
│           ├── orats_qc.py
│           ├── report.py
│           ├── runners.py
│           ├── summarizers.py
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
│   ├── contracts.py
│   ├── greeks.py
│   └── pricing.py
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
```