import pytest

from app.domain.market_rules import ISEMMarketRules
from app.domain.models import DispatchMode, DispatchRecommendation, QuantileForecast


def test_quantile_forecast_validates_ordering() -> None:
    QuantileForecast(point=10.0, p10=8.0, p50=10.0, p90=12.0)
    with pytest.raises(ValueError):
        QuantileForecast(point=10.0, p10=11.0, p50=10.0, p90=12.0)


def test_dispatch_recommendation_confidence_bounds() -> None:
    DispatchRecommendation(
        action=DispatchMode.IDLE,
        setpoint_mw=0.0,
        confidence=0.6,
        expected_impact_eur=0.0,
    )
    with pytest.raises(ValueError):
        DispatchRecommendation(
            action=DispatchMode.IDLE,
            setpoint_mw=0.0,
            confidence=1.2,
            expected_impact_eur=0.0,
        )


def test_isem_rules_validate_supported_interval() -> None:
    rules = ISEMMarketRules()
    rules.validate_interval_minutes(15)
    with pytest.raises(ValueError):
        rules.validate_interval_minutes(7)
