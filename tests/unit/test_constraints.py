from app.domain.constraints import (
    clamp_soc_pct,
    energy_limited_charge_mw,
    energy_limited_discharge_mw,
    project_soc_pct,
)
from app.domain.models import BatteryState


def test_clamp_soc_pct_applies_bounds() -> None:
    assert clamp_soc_pct(-5.0) == 0.0
    assert clamp_soc_pct(55.0) == 55.0
    assert clamp_soc_pct(120.0) == 100.0


def test_energy_limited_charge_respects_headroom_and_power_limit() -> None:
    battery = BatteryState(soc_pct=90.0, capacity_mwh=10.0, max_charge_mw=8.0, max_discharge_mw=8.0)
    # 10% headroom on 10 MWh = 1 MWh; over 15m => 4 MW max from energy limit
    assert energy_limited_charge_mw(battery, interval_minutes=15) == 4.0


def test_energy_limited_discharge_respects_soc_floor_and_power_limit() -> None:
    battery = BatteryState(soc_pct=20.0, capacity_mwh=10.0, max_charge_mw=8.0, max_discharge_mw=20.0)
    # 20% available on 10 MWh = 2 MWh; over 30m => 4 MW max from energy limit
    assert energy_limited_discharge_mw(battery, interval_minutes=30) == 4.0


def test_project_soc_pct_charging_and_discharging() -> None:
    battery = BatteryState(soc_pct=50.0, capacity_mwh=4.0, max_charge_mw=4.0, max_discharge_mw=4.0)

    # +2 MW for 30m = +1 MWh => +25%
    assert project_soc_pct(battery, setpoint_mw=2.0, interval_minutes=30) == 75.0
    # -4 MW for 15m = -1 MWh => -25%
    assert project_soc_pct(battery, setpoint_mw=-4.0, interval_minutes=15) == 25.0
