# API Contract (v1 draft)

## Endpoint

`GET /v1/forecasts/{participant_id}`

## Per-Interval Fields

- `imbalanceForecastMw` (point + quantiles)
- `priceForecastEurMwh` (point + quantiles)
- `priceSpikeProbability` (`0..1`)
- `dispatchRecommendation`
  - action: `CHARGE | DISCHARGE | IDLE`
  - setpoint MW
  - confidence (`0..1`)
  - expected impact EUR
- `socForecastPct` (`0..100`)

## Quality Flags

- freshness/staleness
- fallback mode indicator
