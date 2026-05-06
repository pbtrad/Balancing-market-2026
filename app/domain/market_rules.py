"""I-SEM-specific rule configuration isolated from generic logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ISEMMarketRules:
    """Single location for market-specific assumptions and validation."""

    default_interval_minutes: int = 15
    supported_intervals: tuple[int, ...] = (5, 15, 30, 60)
    high_price_spike_threshold_eur_mwh: float = 250.0

    def validate_interval_minutes(self, interval_minutes: int) -> None:
        if interval_minutes not in self.supported_intervals:
            raise ValueError(
                f"Unsupported interval {interval_minutes}. "
                f"Supported: {self.supported_intervals}"
            )
