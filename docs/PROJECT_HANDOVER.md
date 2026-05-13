# PROJECT_HANDOVER.md

## Project Intent

Build a new **Energy Decision Engine** for the **Integrated Single Electricity Market (I-SEM)** on an all-island basis (Republic of Ireland + Northern Ireland), for BM participants who need to:

- predict short-term imbalance,
- anticipate price spikes,
- decide when to dispatch flexible assets (charge/discharge/idle).

This is a **new project from scratch**. The old repository is reference material only.

---

## What We Learned From the Old Project


### Reusable Patterns
- Forecaster loop and scheduling pattern (`app/forecaster.py`)
- Battery constraint math: SOC bounds, charge/discharge caps (`app/forecaster.py`)
- Data prep structure and time feature generation (`data_preparation/get_data.py`)
- Unit test style and edge-case coverage (`tests/*`)
- Simple baseline model concept (`forecast_models/previous_day_model.py`)


---

## I-SEM Context to Respect

From the quick guide and discussion:

- I-SEM is all-island and includes DAM, IDM, BM interactions.
- Participant net position evolves from ex-ante trades and balancing actions.
- Near-real-time decision support requires:
  - imbalance risk forecast,
  - price/spike forecast,
  - constrained dispatch recommendation.

---

## New Project Architecture

```text
energy-decision-engine/
├── app/
│   ├── domain/
│   │   ├── models.py
│   │   ├── constraints.py
│   │   ├── dispatch.py
│   │   └── market_rules.py
│   ├── services/
│   │   ├── forecasting_service.py
│   │   └── dispatch_service.py
│   ├── ports/
│   │   ├── market_data_port.py
│   │   └── model_port.py
│   └── config.py
├── infrastructure/
│   ├── data_sources/
│   │   ├── semo_client.py
│   │   ├── eirgrid_client.py
│   │   └── weather_client.py
│   ├── storage/
│   │   ├── s3.py
│   │   └── dynamodb.py
│   └── config.py
├── pipelines/
│   ├── ingestion.py
│   ├── feature_engineering.py
│   └── dataset_builder.py
├── ml/
│   ├── train/
│   ├── inference/
│   ├── baseline/
│   └── artifacts/
├── api/
│   ├── schemas.py
│   └── routes.py
├── backtesting/
│   ├── backtest.py
│   └── metrics.py
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── market_assumptions.md
│   └── api_contract.md
└── README.md
```

---

## Target API Contract (v1 draft)

Endpoint: `GET /v1/forecasts/{participant_id}`

### Per-interval outputs (units explicit)
- `imbalanceForecastMw` (point + quantiles)
- `priceForecastEurMwh` (point + quantiles)
- `priceSpikeProbability` (0-1)
- `dispatchRecommendation` (`CHARGE`/`DISCHARGE`/`IDLE`, setpoint MW, confidence, expected impact EUR)
- `socForecastPct`

### Quality flags
- freshness/staleness
- fallback mode indicator

## Time/Cadence Assumptions

- Internal timestamps in UTC.
- Operational timezone handling for Europe/Dublin where needed.
- Base interval configurable (target 15-min unless changed explicitly).
- Horizons: at least 2h + 24h.

## Delivery Plan (phased)

### Phase 1: Foundation
- Scaffold repo structure.
- Define docs (`market_assumptions.md`, `api_contract.md`).
- Create domain models and constraints with tests.
- Implement baseline previous-day model in `ml/baseline`.

### Phase 2: Data and Features
- Build ingestion adapters (SEMO/EirGrid placeholders first).
- Implement feature engineering (lags, ramps, intraday features, holiday handling ROI+NI).
- Add schema validation for raw feeds.

### Phase 3: Forecasting
- Implement imbalance model inference path.
- Implement price model + spike probability path.
- Add quantile outputs and confidence handling.

### Phase 4: Dispatch
- Build dispatch decision logic with constraints.
- Compute expected impact and confidence.
- Add fallback policy when data/model confidence is poor.

### Phase 5: API + Integration
- Expose `/v1/forecasts/{participant_id}`.
- Add integration tests for contract and behavior.
- Add quality/fallback flags in response.

### Phase 6: Backtesting + Hardening
- Rolling backtests for imbalance/price/spikes.
- Compare vs baseline.
- Add monitoring hooks and runbook.

## Old-to-New Mapping (reference-only migration)

### Old `app/forecaster.py`
- Reuse: constraint/capping/SOC logic ideas
- Replace: tariff-based cost and price logic

### Old `data_preparation/get_data.py`
- Reuse: feature pipeline pattern and missing-data handling concepts
- Replace: country/holiday assumptions and cadence logic

### Old `forecast_models/previous_day_model.py`
- Reuse: as baseline fallback only

### Old `tests/*`
- Reuse: test style and edge-case strategy
- Replace: market-specific expected values

## Non-Goals (for v1)

- Full settlement engine replication
- Non-energy ancillary optimization
- Full production infra on day 1

## Risks to Track

- Data quality and timeliness from external feeds
- Rare-event spike model overfitting
- Ambiguity in target labels for imbalance/price
- Operational trust in dispatch recommendations

## Definition of Done (v1)

- End-to-end pipeline runs locally.
- API returns validated contract for a participant.
- Forecast + dispatch outputs are generated for configured horizon.
- Backtest shows improvement over baseline on agreed metrics.
- Fallback mode works when data/model unavailable.
