"""Dispatch service orchestration placeholder for Phase 1."""

from dataclasses import dataclass

from app.domain.dispatch import recommend_dispatch
from app.domain.models import BatteryState, DispatchRecommendation


@dataclass
class DispatchService:
    """Coordinates dispatch recommendations."""

    def run(
        self,
        battery: BatteryState,
        interval_minutes: int,
        expected_price_eur_mwh: float,
        spike_probability: float,
    ) -> DispatchRecommendation:
        return recommend_dispatch(
            battery=battery,
            interval_minutes=interval_minutes,
            expected_price_eur_mwh=expected_price_eur_mwh,
            spike_probability=spike_probability,
        )
