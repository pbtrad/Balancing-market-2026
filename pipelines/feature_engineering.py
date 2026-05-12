"""
Feature engineering pipeline for the Energy Decision Engine.

Transforms validated, normalised records from the ingestion layer into
model-ready feature vectors suitable for imbalance and price forecasting.

Design principles:
- All transformations are stateless and deterministic given the same input.
- Features are grouped by family (lag, ramp, temporal, rolling, calendar)
  so they can be enabled/disabled independently per model.
- No silent NaN propagation: every transformation documents its null
  behaviour and the pipeline surface missing values explicitly.
- Holiday calendars for both ROI and NI are handled from a single entry
  point so market-boundary differences never leak into model code.
- Output is a FeatureFrame: a lightweight typed container that carries
  both the feature matrix and a feature manifest for auditability.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeatureFamily(str, Enum):
    LAG = "lag"
    RAMP = "ramp"
    TEMPORAL = "temporal"
    ROLLING = "rolling"
    CALENDAR = "calendar"


# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntervalRecord:
    """
    One normalised market interval as produced by the ingestion layer.

    All values use explicit units in the field name.
    Missing optional values are represented as None, never as sentinels.
    """

    timestamp_utc: datetime
    price_eur_mwh: float | None
    imbalance_mw: float | None
    system_demand_mw: float | None
    wind_generation_mw: float | None
    temperature_celsius: float | None

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware.")


@dataclass
class FeatureVector:
    """
    All engineered features for a single interval.

    Features absent due to insufficient history are stored as None.
    The consumer (model training / inference) decides how to handle them.
    """

    timestamp_utc: datetime
    features: dict[str, float | None]
    missing_count: int = 0

    def __post_init__(self) -> None:
        self.missing_count = sum(1 for v in self.features.values() if v is None)


@dataclass
class FeatureManifest:
    """
    Audit record describing every feature in the output.

    Stored alongside training datasets so feature definitions are
    always reproducible.
    """

    feature_name: str
    family: FeatureFamily
    description: str
    unit: str | None = None
    nullable: bool = True


@dataclass
class FeatureFrame:
    """
    Container returned by the pipeline: feature vectors + manifest.

    The manifest lists every feature that was attempted; the vectors
    contain the computed values (None where unavailable).
    """

    vectors: list[FeatureVector]
    manifest: list[FeatureManifest]
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def feature_names(self) -> list[str]:
        return [m.feature_name for m in self.manifest]

    @property
    def total_intervals(self) -> int:
        return len(self.vectors)

    @property
    def overall_missing_rate(self) -> float:
        if not self.vectors or not self.manifest:
            return 0.0
        total_cells = len(self.vectors) * len(self.manifest)
        missing_cells = sum(v.missing_count for v in self.vectors)
        return missing_cells / total_cells


# ---------------------------------------------------------------------------
# Holiday calendars — ROI and NI kept separate
# ---------------------------------------------------------------------------

def _roi_public_holidays(year: int) -> frozenset[date]:
    """
    Republic of Ireland public holidays for a given year.

    Fixed-date and rule-based holidays only.  Easter-dependent holidays
    use the anonymous Gregorian algorithm (no external dependency).
    """
    easter = _easter_sunday(year)
    return frozenset({
        date(year, 1, 1),                        # New Year's Day
        date(year, 2, 3) if date(year, 2, 3).weekday() == 0  # St Brigid's (first Mon Feb)
            else _first_monday_in_month(year, 2),
        easter - timedelta(days=2),              # Good Friday (not statutory but observed)
        easter,                                  # Easter Sunday
        easter + timedelta(days=1),              # Easter Monday
        _first_monday_in_month(year, 5),         # May Bank Holiday
        _first_monday_in_month(year, 6),         # June Bank Holiday
        _first_monday_in_month(year, 8),         # August Bank Holiday
        _last_monday_in_month(year, 10),         # October Bank Holiday
        date(year, 12, 25),                      # Christmas Day
        date(year, 12, 26),                      # St Stephen's Day
    })


def _ni_public_holidays(year: int) -> frozenset[date]:
    """
    Northern Ireland public holidays for a given year.

    NI follows the UK bank holiday schedule with two NI-specific additions:
    St Patrick's Day and the Battle of the Boyne.
    """
    easter = _easter_sunday(year)
    may_day = _first_monday_in_month(year, 5)
    # UK Spring Bank Holiday: last Monday in May
    spring_bh = _last_monday_in_month(year, 5)
    # UK Summer Bank Holiday: last Monday in August
    summer_bh = _last_monday_in_month(year, 8)

    return frozenset({
        date(year, 1, 1),                        # New Year's Day
        date(year, 3, 17),                       # St Patrick's Day (NI only)
        easter - timedelta(days=2),              # Good Friday
        easter + timedelta(days=1),              # Easter Monday
        may_day,                                 # Early May Bank Holiday
        spring_bh,                               # Spring Bank Holiday
        date(year, 7, 12),                       # Battle of the Boyne (NI only)
        summer_bh,                               # Summer Bank Holiday
        date(year, 12, 25),                      # Christmas Day
        date(year, 12, 26),                      # Boxing Day
    })


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(114 + h + l - 7 * m, 31)
    return date(year, month, day + 1)


def _first_monday_in_month(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _last_monday_in_month(year: int, month: int) -> date:
    # Find the last day of the month then walk back to Monday.
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - 0) % 7)


@dataclass(frozen=True)
class AllIslandCalendar:
    """
    Combined ROI + NI calendar.

    Exposes per-jurisdiction flags so models can treat them independently
    or together depending on the participant's grid connection.
    """

    _cache: dict[int, tuple[frozenset[date], frozenset[date]]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def _holidays(self, year: int) -> tuple[frozenset[date], frozenset[date]]:
        if year not in self._cache:
            self._cache[year] = (
                _roi_public_holidays(year),
                _ni_public_holidays(year),
            )
        return self._cache[year]

    def is_roi_holiday(self, d: date) -> bool:
        roi, _ = self._holidays(d.year)
        return d in roi

    def is_ni_holiday(self, d: date) -> bool:
        _, ni = self._holidays(d.year)
        return d in ni

    def is_any_holiday(self, d: date) -> bool:
        return self.is_roi_holiday(d) or self.is_ni_holiday(d)


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """
    Controls which feature families are computed and their parameters.

    All defaults are tuned for 15-minute I-SEM intervals.
    """

    # Lag windows in number of intervals
    lag_intervals: tuple[int, ...] = (1, 2, 4, 8, 96, 192, 672)
    # Ramp windows (delta over N intervals)
    ramp_intervals: tuple[int, ...] = (1, 4, 8)
    # Rolling statistic windows in intervals
    rolling_windows: tuple[int, ...] = (4, 16, 96)   # ~1h, ~4h, ~24h at 15-min
    # Enable/disable families
    enable_lag: bool = True
    enable_ramp: bool = True
    enable_temporal: bool = True
    enable_rolling: bool = True
    enable_calendar: bool = True
    # Rejection threshold: vectors with more missing features than this
    # fraction are logged as warnings.
    missing_rate_warn_threshold: float = 0.3


# ---------------------------------------------------------------------------
# Individual feature families
# ---------------------------------------------------------------------------

def _lag_features(
    records: list[IntervalRecord],
    idx: int,
    config: FeatureConfig,
) -> dict[str, float | None]:
    """
    For each configured lag window, return the value N intervals ago.

    Covers price, imbalance, demand, and wind.
    """
    features: dict[str, float | None] = {}
    targets = {
        "price_eur_mwh": lambda r: r.price_eur_mwh,
        "imbalance_mw": lambda r: r.imbalance_mw,
        "system_demand_mw": lambda r: r.system_demand_mw,
        "wind_generation_mw": lambda r: r.wind_generation_mw,
    }
    for lag in config.lag_intervals:
        src_idx = idx - lag
        for col, getter in targets.items():
            key = f"{col}__lag_{lag}"
            features[key] = getter(records[src_idx]) if src_idx >= 0 else None
    return features


def _ramp_features(
    records: list[IntervalRecord],
    idx: int,
    config: FeatureConfig,
) -> dict[str, float | None]:
    """
    Rate of change: current value minus value N intervals ago.

    Positive ramp = rising; negative = falling.
    """
    features: dict[str, float | None] = {}
    targets = {
        "price_eur_mwh": lambda r: r.price_eur_mwh,
        "imbalance_mw": lambda r: r.imbalance_mw,
    }
    current = records[idx]
    for window in config.ramp_intervals:
        src_idx = idx - window
        for col, getter in targets.items():
            key = f"{col}__ramp_{window}"
            if src_idx >= 0:
                past_val = getter(records[src_idx])
                curr_val = getter(current)
                features[key] = (
                    curr_val - past_val
                    if curr_val is not None and past_val is not None
                    else None
                )
            else:
                features[key] = None
    return features


def _temporal_features(ts: datetime) -> dict[str, float | None]:
    """
    Cyclical encoding of time-of-day, day-of-week, and month.

    Raw integers (hour=14) are poor model inputs because hour 23 and hour 0
    are adjacent in time but far apart numerically.  Sin/cos encoding
    preserves cyclical continuity.
    """
    local = ts  # Caller is responsible for tz conversion if needed.
    interval_of_day = ts.hour * 4 + ts.minute // 15  # 0..95 for 15-min grid
    dow = ts.weekday()  # 0=Monday
    month = ts.month

    def encode(value: float, period: float) -> tuple[float, float]:
        angle = 2 * math.pi * value / period
        return math.sin(angle), math.cos(angle)

    tod_sin, tod_cos = encode(interval_of_day, 96)
    dow_sin, dow_cos = encode(dow, 7)
    month_sin, month_cos = encode(month - 1, 12)

    return {
        "temporal__interval_of_day": float(interval_of_day),
        "temporal__tod_sin": tod_sin,
        "temporal__tod_cos": tod_cos,
        "temporal__dow": float(dow),
        "temporal__dow_sin": dow_sin,
        "temporal__dow_cos": dow_cos,
        "temporal__month_sin": month_sin,
        "temporal__month_cos": month_cos,
        "temporal__is_weekend": float(dow >= 5),
    }


def _rolling_features(
    records: list[IntervalRecord],
    idx: int,
    config: FeatureConfig,
) -> dict[str, float | None]:
    """
    Rolling mean and standard deviation over configured windows.

    Window is left-closed [idx-window+1 .. idx] inclusive.
    Returns None if the window cannot be fully populated.
    """
    features: dict[str, float | None] = {}
    targets = {
        "price_eur_mwh": lambda r: r.price_eur_mwh,
        "imbalance_mw": lambda r: r.imbalance_mw,
        "wind_generation_mw": lambda r: r.wind_generation_mw,
    }
    for window in config.rolling_windows:
        start = idx - window + 1
        slice_ = records[max(0, start): idx + 1] if start >= 0 else []
        for col, getter in targets.items():
            vals = [getter(r) for r in slice_ if getter(r) is not None]
            if len(vals) < window:
                features[f"{col}__roll_mean_{window}"] = None
                features[f"{col}__roll_std_{window}"] = None
            else:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                features[f"{col}__roll_mean_{window}"] = mean
                features[f"{col}__roll_std_{window}"] = math.sqrt(variance)
    return features


def _calendar_features(
    ts: datetime,
    calendar: AllIslandCalendar,
) -> dict[str, float | None]:
    """
    Binary flags for ROI holiday, NI holiday, and combined.

    Kept as separate flags so a participant in NI and one in ROI
    can both be served from the same feature set.
    """
    d = ts.date()
    return {
        "calendar__is_roi_holiday": float(calendar.is_roi_holiday(d)),
        "calendar__is_ni_holiday": float(calendar.is_ni_holiday(d)),
        "calendar__is_any_holiday": float(calendar.is_any_holiday(d)),
    }


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def _build_manifest(config: FeatureConfig) -> list[FeatureManifest]:
    """
    Produce the full feature manifest matching the configured feature set.

    The manifest is generated from config, not from actual data, so it
    is available before any records are processed.
    """
    manifests: list[FeatureManifest] = []
    signal_units = {
        "price_eur_mwh": "EUR/MWh",
        "imbalance_mw": "MW",
        "system_demand_mw": "MW",
        "wind_generation_mw": "MW",
    }

    if config.enable_lag:
        for lag in config.lag_intervals:
            for col, unit in signal_units.items():
                manifests.append(FeatureManifest(
                    feature_name=f"{col}__lag_{lag}",
                    family=FeatureFamily.LAG,
                    description=f"{col} lagged by {lag} intervals",
                    unit=unit,
                ))

    if config.enable_ramp:
        for window in config.ramp_intervals:
            for col in ("price_eur_mwh", "imbalance_mw"):
                manifests.append(FeatureManifest(
                    feature_name=f"{col}__ramp_{window}",
                    family=FeatureFamily.RAMP,
                    description=f"Change in {col} over {window} intervals",
                    unit=signal_units[col],
                ))

    if config.enable_temporal:
        for name, desc in [
            ("temporal__interval_of_day", "Interval index within day (0..95)"),
            ("temporal__tod_sin", "Sine encoding of time of day"),
            ("temporal__tod_cos", "Cosine encoding of time of day"),
            ("temporal__dow", "Day of week (0=Monday)"),
            ("temporal__dow_sin", "Sine encoding of day of week"),
            ("temporal__dow_cos", "Cosine encoding of day of week"),
            ("temporal__month_sin", "Sine encoding of month"),
            ("temporal__month_cos", "Cosine encoding of month"),
            ("temporal__is_weekend", "1 if Saturday or Sunday"),
        ]:
            manifests.append(FeatureManifest(
                feature_name=name,
                family=FeatureFamily.TEMPORAL,
                description=desc,
                nullable=False,
            ))

    if config.enable_rolling:
        for window in config.rolling_windows:
            for col, unit in {
                "price_eur_mwh": "EUR/MWh",
                "imbalance_mw": "MW",
                "wind_generation_mw": "MW",
            }.items():
                manifests.append(FeatureManifest(
                    feature_name=f"{col}__roll_mean_{window}",
                    family=FeatureFamily.ROLLING,
                    description=f"Rolling mean of {col} over {window} intervals",
                    unit=unit,
                ))
                manifests.append(FeatureManifest(
                    feature_name=f"{col}__roll_std_{window}",
                    family=FeatureFamily.ROLLING,
                    description=f"Rolling std dev of {col} over {window} intervals",
                    unit=unit,
                ))

    if config.enable_calendar:
        for name, desc in [
            ("calendar__is_roi_holiday", "1 if ROI public holiday"),
            ("calendar__is_ni_holiday", "1 if NI public holiday"),
            ("calendar__is_any_holiday", "1 if ROI or NI public holiday"),
        ]:
            manifests.append(FeatureManifest(
                feature_name=name,
                family=FeatureFamily.CALENDAR,
                description=desc,
                nullable=False,
            ))

    return manifests


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FeatureEngineeringPipeline:
    """
    Transforms a sequence of IntervalRecords into a FeatureFrame.

    Records must be sorted in ascending timestamp order.  Gaps in the
    series are tolerated: missing values propagate as None rather than
    causing failure, and are surfaced via FeatureVector.missing_count.

    Usage::

        pipeline = FeatureEngineeringPipeline(config=FeatureConfig())
        frame = pipeline.run(records)
        # frame.vectors  — one FeatureVector per input record
        # frame.manifest — full feature definition list
    """

    config: FeatureConfig = field(default_factory=FeatureConfig)
    calendar: AllIslandCalendar = field(default_factory=AllIslandCalendar)

    def run(self, records: Sequence[IntervalRecord]) -> FeatureFrame:
        """
        Engineer features for every interval in the input sequence.

        Args:
            records: Normalised market intervals, ascending timestamp order.

        Returns:
            FeatureFrame containing one FeatureVector per record.

        Raises:
            ValueError: if records are empty or not sorted.
        """
        records = list(records)
        self._validate_inputs(records)

        manifest = _build_manifest(self.config)
        vectors: list[FeatureVector] = []

        for idx, record in enumerate(records):
            features: dict[str, float | None] = {}

            if self.config.enable_lag:
                features.update(_lag_features(records, idx, self.config))

            if self.config.enable_ramp:
                features.update(_ramp_features(records, idx, self.config))

            if self.config.enable_temporal:
                features.update(_temporal_features(record.timestamp_utc))

            if self.config.enable_rolling:
                features.update(_rolling_features(records, idx, self.config))

            if self.config.enable_calendar:
                features.update(_calendar_features(record.timestamp_utc, self.calendar))

            vector = FeatureVector(
                timestamp_utc=record.timestamp_utc,
                features=features,
            )

            if vector.missing_count / max(len(features), 1) > self.config.missing_rate_warn_threshold:
                logger.warning(
                    "High missing rate at %s: %d/%d features absent.",
                    record.timestamp_utc.isoformat(),
                    vector.missing_count,
                    len(features),
                )

            vectors.append(vector)

        frame = FeatureFrame(vectors=vectors, manifest=manifest)

        logger.info(
            "Feature engineering complete: %d intervals, %d features, "
            "overall missing rate %.2f%%.",
            frame.total_intervals,
            len(frame.manifest),
            frame.overall_missing_rate * 100,
        )

        return frame

    @staticmethod
    def _validate_inputs(records: list[IntervalRecord]) -> None:
        if not records:
            raise ValueError("records cannot be empty.")
        for i in range(1, len(records)):
            if records[i].timestamp_utc <= records[i - 1].timestamp_utc:
                raise ValueError(
                    f"Records must be in ascending timestamp order. "
                    f"Found {records[i].timestamp_utc!r} after "
                    f"{records[i - 1].timestamp_utc!r} at index {i}."
                )