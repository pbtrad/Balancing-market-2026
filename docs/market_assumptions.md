# Market Assumptions (Phase 1)

## Scope

This document captures initial assumptions for a market-aware but modular v1 foundation.

## Core Assumptions

- Market context is I-SEM (all-island), but base forecasting/constraint logic remains market-agnostic.
- Internal timestamps are UTC; conversion to Europe/Dublin is an operational concern at system edges.
- Base interval defaults to 15 minutes and must remain configurable.
- Forecast horizons should support both near-term and day-ahead views.

## Isolated I-SEM Rules

I-SEM-specific parameters are isolated in `app/domain/market_rules.py` so core algorithms can be reused for other markets with minimal code changes.
