"""Forecast model port definitions."""

from typing import Protocol


class ForecastModelPort(Protocol):
    """Forecast model interface for application services."""

    def forecast(self, values: list[float], horizon_steps: int, steps_per_day: int) -> list[float]:
        """Return forecast values for the requested horizon."""
