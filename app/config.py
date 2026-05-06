"""App-level configuration defaults for Phase 1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration shared by app services."""

    interval_minutes: int = 15
    short_horizon_minutes: int = 120
    long_horizon_minutes: int = 24 * 60
