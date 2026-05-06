"""Simple previous-day baseline forecast model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreviousDayBaselineModel:
    """
    Repeat value from one day ago at same interval slot.

    If a lookback point is missing, fallback to the most recent value.
    """

    fallback_to_last_observation: bool = True

    def forecast(self, values: list[float], horizon_steps: int, steps_per_day: int) -> list[float]:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        if not values:
            raise ValueError("values cannot be empty")

        forecasts: list[float] = []
        n = len(values)
        for i in range(horizon_steps):
            lookup_index = (n - steps_per_day) + i
            if 0 <= lookup_index < n:
                forecasts.append(values[lookup_index])
            elif self.fallback_to_last_observation:
                forecasts.append(values[-1])
            else:
                raise ValueError("insufficient history for previous-day lookup")
        return forecasts
