"""Constraint math for storage dispatch and SOC evolution."""

from app.domain.models import BatteryState


def clamp_soc_pct(soc_pct: float, min_soc_pct: float = 0.0, max_soc_pct: float = 100.0) -> float:
    """Clamp SOC percentage into configured bounds."""
    if min_soc_pct > max_soc_pct:
        raise ValueError("min_soc_pct cannot exceed max_soc_pct")
    return max(min_soc_pct, min(max_soc_pct, soc_pct))


def energy_limited_charge_mw(
    battery: BatteryState,
    interval_minutes: int,
    target_max_soc_pct: float = 100.0,
) -> float:
    """Return max feasible charge MW constrained by headroom and power cap."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    headroom_pct = max(0.0, target_max_soc_pct - battery.soc_pct)
    headroom_mwh = battery.capacity_mwh * (headroom_pct / 100.0)
    interval_hours = interval_minutes / 60.0
    energy_limited_mw = headroom_mwh / interval_hours if interval_hours > 0 else 0.0
    return max(0.0, min(battery.max_charge_mw, energy_limited_mw))


def energy_limited_discharge_mw(
    battery: BatteryState,
    interval_minutes: int,
    target_min_soc_pct: float = 0.0,
) -> float:
    """Return max feasible discharge MW constrained by SOC floor and power cap."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    available_pct = max(0.0, battery.soc_pct - target_min_soc_pct)
    available_mwh = battery.capacity_mwh * (available_pct / 100.0)
    interval_hours = interval_minutes / 60.0
    energy_limited_mw = available_mwh / interval_hours if interval_hours > 0 else 0.0
    return max(0.0, min(battery.max_discharge_mw, energy_limited_mw))


def project_soc_pct(
    battery: BatteryState,
    setpoint_mw: float,
    interval_minutes: int,
    min_soc_pct: float = 0.0,
    max_soc_pct: float = 100.0,
) -> float:
    """
    Project next SOC based on setpoint convention:
    - positive setpoint: charging
    - negative setpoint: discharging
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    delta_mwh = setpoint_mw * (interval_minutes / 60.0)
    delta_pct = (delta_mwh / battery.capacity_mwh) * 100.0
    return clamp_soc_pct(battery.soc_pct + delta_pct, min_soc_pct=min_soc_pct, max_soc_pct=max_soc_pct)
