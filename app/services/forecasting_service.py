"""Forecasting service orchestration placeholder for Phase 1."""

from dataclasses import dataclass

from app.ports.model_port import ForecastModelPort


@dataclass
class ForecastingService:
    """Orchestrates forecasting via model ports."""

    model: ForecastModelPort

    def run(self, values: list[float], horizon_steps: int, steps_per_day: int) -> list[float]:
        return self.model.forecast(values=values, horizon_steps=horizon_steps, steps_per_day=steps_per_day)
