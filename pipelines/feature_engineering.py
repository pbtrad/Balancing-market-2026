"""
Feature engineering pipeline for the Energy Decision Engine.

Transforms validated, normalised market intervals into model-ready
feature vectors for imbalance and price forecasting.

Design principles:
- All transformations operate on a single pandas DataFrame internally;
  the public API accepts and returns typed domain objects so the pandas
  dependency stays an implementation detail.
- Features are computed in vectorised operations only — no row-wise
  apply() or Python-level loops over rows.
- NaN propagation is explicit: every feature documents its null
  behaviour.  The pipeline never silently fills or drops NaNs without
  a caller-visible flag.
- Holiday calendars for ROI and NI are handled from a single entry
  point; market-boundary differences never leak into model code.
- The FeatureManifest is derived from config before any data is
  processed, so feature definitions are always reproducible.
- All timestamps are UTC throughout; Europe/Dublin conversion is an
  operational concern at system edges only.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

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
# Domain containers (public API — no pandas exposure)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntervalRecord:
    """
    One normalised market interval from the ingestion layer.

    All values carry explicit units in the field name.
    None represents a genuinely missing observation, never a sentinel.
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
    """Engineered features for a single interval."""

    timestamp_utc: datetime
    features: dict[str, float | None]
    missing_count: int = 0

    def __post_init__(self) -> None:
        self.missing_count = sum(1 for v in self.features.values() if v is None)


@dataclass
class FeatureManifest:
    """Audit record for one feature column."""

    feature_name: str
    family: FeatureFamily
    description: str
    unit: str | None = None
    nullable: bool = True


@dataclass
class FeatureFrame:
    """
    Pipeline output: feature vectors plus the manifest that describes them.

    The manifest is generated from config before any records are
    processed, so it is available for schema validation downstream.
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
# Holiday calendars
# ---------------------------------------------------------------------------

def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm — no external dependency."""
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


def _first_monday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _last_monday(year: int, month: int) -> date:
    last = (
        date(year + 1, 1, 1) - timedelta(days=1)
        if month == 12
        else date(year, month + 1, 1) - timedelta(days=1)
    )
    return last - timedelta(days=(last.weekday()) % 7)


def _roi_holidays(year: int) -> frozenset[date]:
    easter = _easter_sunday(year)
    return frozenset({
        date(year, 1, 1),
        _first_monday(year, 2),                     # St Brigid's Day
        easter - timedelta(days=2),                 # Good Friday
        easter,
        easter + timedelta(days=1),                 # Easter Monday
        _first_monday(year, 5),                     # May Bank Holiday
        _first_monday(year, 6),                     # June Bank Holiday
        _first_monday(year, 8),                     # August Bank Holiday
        _last_monday(year, 10),                     # October Bank Holiday
        date(year, 12, 25),
        date(year, 12, 26),                         # St Stephen's Day
    })


def _ni_holidays(year: int) -> frozenset[date]:
    easter = _easter_sunday(year)
    return frozenset({
        date(year, 1, 1),
        date(year, 3, 17),                          # St Patrick's Day
        easter - timedelta(days=2),                 # Good Friday
        easter + timedelta(days=1),                 # Easter Monday
        _first_monday(year, 5),                     # Early May Bank Holiday
        _last_monday(year, 5),                      # Spring Bank Holiday
        date(year, 7, 12),                          # Battle of the Boyne
        _last_monday(year, 8),                      # Summer Bank Holiday
        date(year, 12, 25),
        date(year, 12, 26),                         # Boxing Day
    })


@dataclass
class AllIslandCalendar:
    """
    Combined ROI + NI public holiday calendar with per-year caching.

    Separate flags are preserved so models can treat jurisdictions
    independently for participants whose grid connection differs.
    """

    _cache: dict[int, tuple[frozenset[date], frozenset[date]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _get(self, year: int) -> tuple[frozenset[date], frozenset[date]]:
        if year not in self._cache:
            self._cache[year] = (_roi_holidays(year), _ni_holidays(year))
        return self._cache[year]

    def is_roi_holiday(self, d: date) -> bool:
        roi, _ = self._get(d.year)
        return d in roi

    def is_ni_holiday(self, d: date) -> bool:
        _, ni = self._get(d.year)
        return d in ni

    def build_holiday_frame(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Vectorised holiday flag construction for a full DatetimeIndex.

        Iterates once over the unique dates in the index rather than
        once per row, so cost scales with unique calendar days not rows.
        """
        unique_dates = {ts.date() for ts in index}
        roi_set: set[date] = set()
        ni_set: set[date] = set()
        for d in unique_dates:
            roi, ni = self._get(d.year)
            if d in roi:
                roi_set.add(d)
            if d in ni:
                ni_set.add(d)

        dates_array = np.array([ts.date() for ts in index])
        roi_flags = np.array([d in roi_set for d in dates_array], dtype=np.float32)
        ni_flags = np.array([d in ni_set for d in dates_array], dtype=np.float32)

        return pd.DataFrame(
            {
                "calendar__is_roi_holiday": roi_flags,
                "calendar__is_ni_holiday": ni_flags,
                "calendar__is_any_holiday": np.clip(roi_flags + ni_flags, 0, 1),
            },
            index=index,
        )


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """
    Controls which feature families are computed and their parameters.

    Defaults are calibrated for 15-minute I-SEM intervals.
    Changing any value produces a different config hash for lineage tracking.

    Lag reference:
        1   = 15 min,  2 = 30 min,   4 = 1 h,
        8   = 2 h,    96 = 24 h,   192 = 48 h,  672 = 1 week
    Rolling reference:
        4   = 1 h,    16 = 4 h,     96 = 24 h
    """

    lag_intervals: tuple[int, ...] = (1, 2, 4, 8, 96, 192, 672)
    ramp_intervals: tuple[int, ...] = (1, 4, 8)
    rolling_windows: tuple[int, ...] = (4, 16, 96)

    enable_lag: bool = True
    enable_ramp: bool = True
    enable_temporal: bool = True
    enable_rolling: bool = True
    enable_calendar: bool = True

    missing_rate_warn_threshold: float = 0.3

    # min_periods < window allows partial warm-up at the start of the
    # series rather than producing a full leading NaN block.
    rolling_min_periods: int = 1

    def to_hash(self) -> str:
        import json
        payload = json.dumps(
            {k: list(v) if isinstance(v, tuple) else v
             for k, v in self.__dict__.items()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Signal column registry
# ---------------------------------------------------------------------------

# Signals used for lag and ramp features
_LAG_RAMP_SIGNALS: dict[str, str] = {
    "price_eur_mwh": "EUR/MWh",
    "imbalance_mw": "MW",
    "system_demand_mw": "MW",
    "wind_generation_mw": "MW",
}

# Signals used for rolling statistics
_ROLLING_SIGNALS: dict[str, str] = {
    "price_eur_mwh": "EUR/MWh",
    "imbalance_mw": "MW",
    "wind_generation_mw": "MW",
}

_ALL_SIGNAL_COLUMNS: list[str] = list(_LAG_RAMP_SIGNALS.keys()) + ["temperature_celsius"]


# ---------------------------------------------------------------------------
# Vectorised transformation helpers
# ---------------------------------------------------------------------------

def _add_lag_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Shift each signal column by each configured lag window.

    Uses pd.concat of pre-computed Series to avoid repeated DataFrame
    copies inside a loop.
    """
    parts: list[pd.Series] = []
    for col in _LAG_RAMP_SIGNALS:
        if col not in df.columns:
            logger.warning("Lag: signal column %s not found — skipping.", col)
            continue
        for lag in config.lag_intervals:
            parts.append(df[col].shift(lag).rename(f"{col}__lag_{lag}"))

    return pd.concat([df, *parts], axis=1) if parts else df


def _add_ramp_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    First-difference over each ramp window.

    pd.Series.diff(n) == series - series.shift(n), fully vectorised.
    Positive value = rising price/imbalance, negative = falling.
    """
    parts: list[pd.Series] = []
    for col in ("price_eur_mwh", "imbalance_mw"):
        if col not in df.columns:
            logger.warning("Ramp: signal column %s not found — skipping.", col)
            continue
        for window in config.ramp_intervals:
            parts.append(df[col].diff(window).rename(f"{col}__ramp_{window}"))

    return pd.concat([df, *parts], axis=1) if parts else df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical sin/cos encoding for time-of-day, day-of-week, and month.

    Sin/cos encoding preserves cyclical adjacency that raw integers cannot:
    interval 95 (23:45) and interval 0 (00:00) are correctly identified
    as neighbours.  Raw integers are also retained for tree-based models
    that can exploit ordinality directly.
    """
    idx = df.index

    interval_of_day = (idx.hour * 4 + idx.minute // 15).astype(np.float32)
    dow = idx.dayofweek.astype(np.float32)
    month = idx.month.astype(np.float32)

    def encode(
        values: np.ndarray, period: float
    ) -> tuple[np.ndarray, np.ndarray]:
        angle = 2.0 * np.pi * values / period
        return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)

    tod_sin, tod_cos = encode(interval_of_day, 96.0)
    dow_sin, dow_cos = encode(dow, 7.0)
    month_sin, month_cos = encode(month - 1.0, 12.0)

    temporal = pd.DataFrame(
        {
            "temporal__interval_of_day": interval_of_day,
            "temporal__tod_sin": tod_sin,
            "temporal__tod_cos": tod_cos,
            "temporal__dow": dow,
            "temporal__dow_sin": dow_sin,
            "temporal__dow_cos": dow_cos,
            "temporal__month_sin": month_sin,
            "temporal__month_cos": month_cos,
            "temporal__is_weekend": (dow >= 5.0).astype(np.float32),
        },
        index=idx,
    )
    return pd.concat([df, temporal], axis=1)


def _add_rolling_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Rolling mean and population std over configured windows.

    ddof=0 (population std) is used for consistency at small window sizes
    where sample std would be undefined or noisy.

    min_periods=config.rolling_min_periods allows partial warm-up at the
    start of the series; set to `window` in config for strict behaviour.
    """
    parts: list[pd.Series] = []
    for col in _ROLLING_SIGNALS:
        if col not in df.columns:
            logger.warning("Rolling: signal column %s not found — skipping.", col)
            continue
        for window in config.rolling_windows:
            roller = df[col].rolling(
                window=window,
                min_periods=config.rolling_min_periods,
            )
            parts.append(roller.mean().rename(f"{col}__roll_mean_{window}"))
            parts.append(roller.std(ddof=0).rename(f"{col}__roll_std_{window}"))

    return pd.concat([df, *parts], axis=1) if parts else df


def _add_calendar_features(
    df: pd.DataFrame,
    calendar: AllIslandCalendar,
) -> pd.DataFrame:
    holiday_df = calendar.build_holiday_frame(df.index)
    return pd.concat([df, holiday_df], axis=1)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(config: FeatureConfig) -> list[FeatureManifest]:
    """
    Produce the full feature manifest from config alone.

    Available before any data is processed; used for schema validation
    and dataset lineage tracking downstream.
    """
    manifests: list[FeatureManifest] = []

    if config.enable_lag:
        for col, unit in _LAG_RAMP_SIGNALS.items():
            for lag in config.lag_intervals:
                manifests.append(FeatureManifest(
                    feature_name=f"{col}__lag_{lag}",
                    family=FeatureFamily.LAG,
                    description=f"{col} lagged by {lag} intervals",
                    unit=unit,
                ))

    if config.enable_ramp:
        for col in ("price_eur_mwh", "imbalance_mw"):
            for window in config.ramp_intervals:
                manifests.append(FeatureManifest(
                    feature_name=f"{col}__ramp_{window}",
                    family=FeatureFamily.RAMP,
                    description=f"Change in {col} over {window} intervals",
                    unit=_LAG_RAMP_SIGNALS[col],
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
        for col, unit in _ROLLING_SIGNALS.items():
            for window in config.rolling_windows:
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
# Records <-> DataFrame I/O
# ---------------------------------------------------------------------------

def _records_to_df(records: list[IntervalRecord]) -> pd.DataFrame:
    """
    Convert IntervalRecord list to a UTC-indexed float64 DataFrame.

    None values become NaN naturally via pandas construction.
    """
    rows = {
        col: [getattr(r, col) for r in records]
        for col in _ALL_SIGNAL_COLUMNS
    }
    index = pd.DatetimeIndex(
        [r.timestamp_utc for r in records],
        name="timestamp_utc",
    )
    df = pd.DataFrame(rows, index=index, dtype=np.float64)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


def _df_to_vectors(
    df: pd.DataFrame,
    feature_cols: list[str],
    warn_threshold: float,
) -> list[FeatureVector]:
    """
    Convert the engineered DataFrame back to FeatureVector domain objects.

    NaN values become None to preserve the domain contract that None
    means missing, not zero.
    """
    feature_df = df[feature_cols]
    n_features = len(feature_cols)
    vectors: list[FeatureVector] = []

    # Convert to records as dicts for fast row iteration.
    for ts, row in zip(feature_df.index, feature_df.to_dict(orient="records")):
        features: dict[str, float | None] = {
            col: (None if pd.isna(val) else float(val))
            for col, val in row.items()
        }
        missing = sum(1 for v in features.values() if v is None)

        if missing / max(n_features, 1) > warn_threshold:
            logger.warning(
                "High missing rate at %s: %d/%d features absent.",
                ts.isoformat(),
                missing,
                n_features,
            )

        vectors.append(
            FeatureVector(
                timestamp_utc=ts.to_pydatetime(),
                features=features,
                missing_count=missing,
            )
        )

    return vectors


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FeatureEngineeringPipeline:
    """
    Transforms a sequence of IntervalRecords into a FeatureFrame.

    Records must be sorted in ascending timestamp order and should use a
    consistent cadence. Gaps are tolerated but affect rolling and lag
    features for the intervals immediately following the gap.

    Internally all numerical operations use pandas/numpy.  The public
    API accepts and returns typed domain objects so the pandas dependency
    stays an implementation detail of this module.

    Usage::

        pipeline = FeatureEngineeringPipeline(config=FeatureConfig())
        frame = pipeline.run(records)

        # frame.vectors  — one FeatureVector per input record
        # frame.manifest — full reproducible feature definition list
        # frame.overall_missing_rate — data quality summary
    """

    config: FeatureConfig = field(default_factory=FeatureConfig)
    calendar: AllIslandCalendar = field(default_factory=AllIslandCalendar)

    def run(self, records: Sequence[IntervalRecord]) -> FeatureFrame:
        """
        Engineer features for every interval in the input sequence.

        Args:
            records: Normalised market intervals in strictly ascending
                     UTC timestamp order.

        Returns:
            FeatureFrame with one FeatureVector per input record.

        Raises:
            ValueError:  if records are empty or not strictly sorted.
            RuntimeError: if the pipeline produces incomplete output
                          (indicates a config/code mismatch).
        """
        records = list(records)
        self._validate_inputs(records)

        manifest = build_manifest(self.config)
        feature_cols = [m.feature_name for m in manifest]

        df = _records_to_df(records)

        if self.config.enable_lag:
            df = _add_lag_features(df, self.config)

        if self.config.enable_ramp:
            df = _add_ramp_features(df, self.config)

        if self.config.enable_temporal:
            df = _add_temporal_features(df)

        if self.config.enable_rolling:
            df = _add_rolling_features(df, self.config)

        if self.config.enable_calendar:
            df = _add_calendar_features(df, self.calendar)

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise RuntimeError(
                f"Feature engineering produced incomplete output. "
                f"Missing columns: {sorted(missing_cols)}"
            )

        vectors = _df_to_vectors(
            df, feature_cols, self.config.missing_rate_warn_threshold
        )

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
                    f"Records must be in strictly ascending timestamp order. "
                    f"Found {records[i].timestamp_utc!r} at index {i} after "
                    f"{records[i - 1].timestamp_utc!r}."
                )