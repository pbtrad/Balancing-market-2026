"""Market data ingestion port definitions."""

from typing import Protocol


class MarketDataPort(Protocol):
    """Interface for reading market time-series inputs."""

    def get_series(self, participant_id: str) -> list[float]:
        """Return participant-specific historic values."""
