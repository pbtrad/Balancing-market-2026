"""Domain models for forecasting and dispatch outputs."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DispatchMode(str, Enum):
    """Supported dispatch decisions for flexible assets."""

    CHARGE = "CHARGE"
    DISCHARGE = "DISCHARGE"
    IDLE = "IDLE"


@dataclass(frozen=True)
class QuantileForecast:
    """Point forecast with optional quantile bounds."""

    point: float
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None

    def __post_init__(self) -> None:
        if self.p10 is not None and self.p50 is not None and self.p10 > self.p50:
            raise ValueError("p10 cannot be greater than p50")
        if self.p50 is not None and self.p90 is not None and self.p50 > self.p90:
            raise ValueError("p50 cannot be greater than p90")


@dataclass(frozen=True)
class DispatchRecommendation:
    """Constrained dispatch recommendation for one interval."""

    action: DispatchMode
    setpoint_mw: float
    confidence: float
    expected_impact_eur: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class BatteryState:
    """Current battery operating state."""

    soc_pct: float
    capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.soc_pct <= 100.0:
            raise ValueError("soc_pct must be in [0, 100]")
        if self.capacity_mwh <= 0:
            raise ValueError("capacity_mwh must be positive")
        if self.max_charge_mw < 0 or self.max_discharge_mw < 0:
            raise ValueError("power limits cannot be negative")


@dataclass(frozen=True)
class ForecastInterval:
    """Per-interval API-aligned forecast output."""

    timestamp_utc: datetime
    imbalance_forecast_mw: QuantileForecast
    price_forecast_eur_mwh: QuantileForecast
    price_spike_probability: float
    dispatch_recommendation: DispatchRecommendation
    soc_forecast_pct: float
    freshness: str = "fresh"
    fallback_mode: bool = False
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.price_spike_probability <= 1.0:
            raise ValueError("price_spike_probability must be in [0, 1]")
        if not 0.0 <= self.soc_forecast_pct <= 100.0:
            raise ValueError("soc_forecast_pct must be in [0, 100]")
