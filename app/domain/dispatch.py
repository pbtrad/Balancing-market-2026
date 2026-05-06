"""Dispatch decision helpers built on domain constraints."""

from app.domain.constraints import energy_limited_charge_mw, energy_limited_discharge_mw
from app.domain.models import BatteryState, DispatchMode, DispatchRecommendation


def recommend_dispatch(
    battery: BatteryState,
    interval_minutes: int,
    expected_price_eur_mwh: float,
    spike_probability: float,
    charge_price_threshold_eur_mwh: float = 40.0,
    discharge_price_threshold_eur_mwh: float = 120.0,
) -> DispatchRecommendation:
    """
    Market-agnostic heuristic:
    - charge in low-price intervals,
    - discharge in high-price/spike-risk intervals,
    - otherwise idle.
    """
    if spike_probability > 0.5 or expected_price_eur_mwh >= discharge_price_threshold_eur_mwh:
        setpoint = -energy_limited_discharge_mw(battery, interval_minutes)
        action = DispatchMode.DISCHARGE if setpoint < 0 else DispatchMode.IDLE
    elif expected_price_eur_mwh <= charge_price_threshold_eur_mwh:
        setpoint = energy_limited_charge_mw(battery, interval_minutes)
        action = DispatchMode.CHARGE if setpoint > 0 else DispatchMode.IDLE
    else:
        setpoint = 0.0
        action = DispatchMode.IDLE

    confidence = min(1.0, max(0.0, 0.4 + abs(expected_price_eur_mwh - 80.0) / 200.0))
    expected_impact = abs(setpoint) * (interval_minutes / 60.0) * expected_price_eur_mwh

    return DispatchRecommendation(
        action=action,
        setpoint_mw=setpoint,
        confidence=confidence,
        expected_impact_eur=expected_impact,
    )
