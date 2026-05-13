"""
Dataset builder pipeline for the Energy Decision Engine.

Joins feature-engineered vectors with target labels to produce
training-ready datasets for imbalance, price, and spike models.

Design principles:
- Target labels are extracted from future IntervalRecords at configured
  horizons; there is no look-ahead into the feature DataFrame itself.
- Train/validation/test splits are strictly chronological with a
  configurable gap to prevent leakage via rolling/lag features.
- Every build produces a DatasetManifest that fully describes the
  construction parameters for reproducibility.
- Missing features and missing targets are handled separately:
  a vector with missing features is kept but warned; a vector where
  every target is missing is excluded from training.
- Internal joins use pandas merge on timestamps for alignment safety —
  never positional index assumptions.
- Output DataFrames are float32 for memory efficiency in training.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from pipelines.feature_engineering import FeatureFrame, IntervalRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetName(str, Enum):
    IMBALANCE_MW = "imbalance_mw"
    PRICE_EUR_MWH = "price_eur_mwh"
    PRICE_SPIKE = "price_spike"          # binary: 1 if price >= spike threshold


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetDefinition:
    """
    Describes how a single target label is derived from a future interval.

    horizon_steps:              intervals ahead the label is read from.
    name:                       which signal is extracted.
    spike_threshold_eur_mwh:    required for PRICE_SPIKE targets only.
    column_name:                derived automatically; used in DataFrames.
    """

    name: TargetName
    horizon_steps: int
    spike_threshold_eur_mwh: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if self.name == TargetName.PRICE_SPIKE and self.spike_threshold_eur_mwh is None:
            raise ValueError(
                "spike_threshold_eur_mwh is required for PRICE_SPIKE target."
            )

    @property
    def column_name(self) -> str:
        return f"{self.name.value}__h{self.horizon_steps}"

    def to_dict(self) -> dict:
        return {
            "name": self.name.value,
            "horizon_steps": self.horizon_steps,
            "spike_threshold_eur_mwh": self.spike_threshold_eur_mwh,
        }


# ---------------------------------------------------------------------------
# Default target suite
# ---------------------------------------------------------------------------

DEFAULT_TARGETS: tuple[TargetDefinition, ...] = (
    TargetDefinition(name=TargetName.IMBALANCE_MW, horizon_steps=1),
    TargetDefinition(name=TargetName.IMBALANCE_MW, horizon_steps=4),
    TargetDefinition(name=TargetName.IMBALANCE_MW, horizon_steps=8),
    TargetDefinition(name=TargetName.PRICE_EUR_MWH, horizon_steps=1),
    TargetDefinition(name=TargetName.PRICE_EUR_MWH, horizon_steps=4),
    TargetDefinition(name=TargetName.PRICE_EUR_MWH, horizon_steps=8),
    TargetDefinition(
        name=TargetName.PRICE_SPIKE,
        horizon_steps=1,
        spike_threshold_eur_mwh=250.0,
    ),
    TargetDefinition(
        name=TargetName.PRICE_SPIKE,
        horizon_steps=4,
        spike_threshold_eur_mwh=250.0,
    ),
)


# ---------------------------------------------------------------------------
# Split policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalSplitPolicy:
    """
    Defines train/validation/test boundaries as fractions of the dataset.

    Splits are strictly chronological — no shuffling ever.

    gap_intervals rows are dropped at each split boundary to prevent
    leakage: a rolling window of 96 intervals (24h) that straddles
    the train/val boundary would otherwise give the validation set
    information from training time.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    gap_intervals: int = 96          # 1 full day at 15-min cadence

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1).")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1).")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError(
                "train_fraction + validation_fraction must sum to less than 1."
            )
        if self.gap_intervals < 0:
            raise ValueError("gap_intervals cannot be negative.")

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction

    def apply(
        self, df: pd.DataFrame
    ) -> dict[SplitName, pd.DataFrame]:
        """
        Partition df into train, validation, test in chronological order.

        The gap is removed from the end of each earlier split, not from
        the start of the later one, so the later split always starts
        at a clean market interval.
        """
        n = len(df)
        train_end = int(n * self.train_fraction)
        val_end = train_end + int(n * self.validation_fraction)

        train_cut = max(0, train_end - self.gap_intervals)
        val_cut = max(train_end, val_end - self.gap_intervals)

        train = df.iloc[:train_cut].copy()
        validation = df.iloc[train_end:val_cut].copy()
        test = df.iloc[val_end:].copy()

        logger.info(
            "Temporal split applied: train=%d, validation=%d, test=%d rows "
            "(gap=%d intervals dropped at each boundary, %d total rows in).",
            len(train), len(validation), len(test),
            self.gap_intervals, n,
        )

        return {
            SplitName.TRAIN: train,
            SplitName.VALIDATION: validation,
            SplitName.TEST: test,
        }


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    """
    Audit record for a full dataset build run.

    Serialise to JSON and store alongside the dataset artifact in S3/GCS
    so every training run has unambiguous provenance.
    """

    dataset_id: str
    built_at: datetime
    config_hash: str
    feature_count: int
    target_definitions: list[dict]
    split_sizes: dict[str, int]
    missing_target_rates: dict[str, float]   # "{split}/{target_col}" -> rate
    missing_feature_rate_mean: float
    window_start_utc: str
    window_end_utc: str
    rows_excluded_all_targets_null: int
    rows_warned_high_missing_features: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset_id": self.dataset_id,
                "built_at": self.built_at.isoformat(),
                "config_hash": self.config_hash,
                "feature_count": self.feature_count,
                "target_definitions": self.target_definitions,
                "split_sizes": self.split_sizes,
                "missing_target_rates": self.missing_target_rates,
                "missing_feature_rate_mean": self.missing_feature_rate_mean,
                "window_start_utc": self.window_start_utc,
                "window_end_utc": self.window_end_utc,
                "rows_excluded_all_targets_null": self.rows_excluded_all_targets_null,
                "rows_warned_high_missing_features": self.rows_warned_high_missing_features,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Storage port
# ---------------------------------------------------------------------------

class DatasetStore(Protocol):
    """Persist split DataFrames and the build manifest."""

    def write_split(
        self,
        dataset_id: str,
        split: SplitName,
        features: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None: ...

    def write_manifest(self, dataset_id: str, manifest: DatasetManifest) -> None: ...


# ---------------------------------------------------------------------------
# Builder config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetBuilderConfig:
    """
    Full configuration for a dataset build.

    Any change produces a different config_hash so datasets built with
    different parameters are distinguishable in storage without parsing
    the full manifest.
    """

    targets: tuple[TargetDefinition, ...] = DEFAULT_TARGETS
    split_policy: TemporalSplitPolicy = field(
        default_factory=TemporalSplitPolicy
    )
    missing_feature_warn_threshold: float = 0.30

    def to_hash(self) -> str:
        payload = json.dumps(
            {
                "targets": [t.to_dict() for t in self.targets],
                "train_fraction": self.split_policy.train_fraction,
                "validation_fraction": self.split_policy.validation_fraction,
                "gap_intervals": self.split_policy.gap_intervals,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_raw_df(raw_records: list[IntervalRecord]) -> pd.DataFrame:
    """
    Convert raw records to a UTC-indexed DataFrame for target extraction.

    Columns match the signal names used by TargetDefinition.extract().
    """
    index = pd.DatetimeIndex(
        [r.timestamp_utc for r in raw_records], name="timestamp_utc"
    )
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    return pd.DataFrame(
        {
            "price_eur_mwh": [r.price_eur_mwh for r in raw_records],
            "imbalance_mw": [r.imbalance_mw for r in raw_records],
        },
        index=index,
        dtype=np.float64,
    )


def _build_feature_df(frame: FeatureFrame) -> pd.DataFrame:
    """
    Convert FeatureFrame to a UTC-indexed float32 DataFrame.

    float32 reduces memory footprint by ~50% vs float64 with negligible
    precision loss for tree-based and neural models.
    """
    index = pd.DatetimeIndex(
        [v.timestamp_utc for v in frame.vectors], name="timestamp_utc"
    )
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    feature_names = frame.feature_names
    data = np.array(
        [
            [
                np.nan if v.features.get(col) is None else v.features[col]
                for col in feature_names
            ]
            for v in frame.vectors
        ],
        dtype=np.float32,
    )
    return pd.DataFrame(data, index=index, columns=feature_names)


def _extract_targets(
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    targets: tuple[TargetDefinition, ...],
) -> pd.DataFrame:
    """
    Align raw_df with feature_df and shift signal columns to produce labels.

    Uses pd.DataFrame.shift() on the raw signal columns so that the
    label at position i is the value at position i + horizon_steps,
    which avoids any positional arithmetic on index values.

    A negative shift moves values backward in the index (i.e. future
    values appear at earlier positions), which is exactly what we need
    for supervised learning with a fixed horizon.
    """
    # Align raw signals onto the feature index via merge.
    # This is timestamp-safe and handles any gap in coverage correctly.
    aligned = feature_df[[]].join(raw_df, how="left")

    target_cols: dict[str, pd.Series] = {}
    for t in targets:
        col = t.column_name
        if t.name == TargetName.IMBALANCE_MW:
            shifted = aligned["imbalance_mw"].shift(-t.horizon_steps)
        elif t.name == TargetName.PRICE_EUR_MWH:
            shifted = aligned["price_eur_mwh"].shift(-t.horizon_steps)
        elif t.name == TargetName.PRICE_SPIKE:
            price = aligned["price_eur_mwh"].shift(-t.horizon_steps)
            shifted = (price >= t.spike_threshold_eur_mwh).astype(np.float32)
            shifted[price.isna()] = np.nan
        else:
            raise ValueError(f"Unknown target: {t.name}")
        target_cols[col] = shifted

    return pd.DataFrame(target_cols, index=feature_df.index, dtype=np.float32)


def _drop_all_targets_null(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Remove rows where every target column is NaN.

    These are intervals at the tail of the series where no future record
    exists for any configured horizon.  They carry no training signal.
    """
    all_null_mask = target_df.isna().all(axis=1)
    excluded = int(all_null_mask.sum())

    if excluded > 0:
        logger.info(
            "Excluding %d rows where all targets are null (tail of series).",
            excluded,
        )

    return (
        feature_df.loc[~all_null_mask],
        target_df.loc[~all_null_mask],
        excluded,
    )


def _count_high_missing_features(
    feature_df: pd.DataFrame,
    threshold: float,
) -> int:
    """
    Count rows whose feature missing rate exceeds the warn threshold.

    These rows are kept in the dataset; the count is reported in the
    manifest for data quality monitoring.
    """
    missing_rate = feature_df.isna().mean(axis=1)
    high_missing = (missing_rate > threshold).sum()
    if high_missing > 0:
        logger.warning(
            "%d rows have feature missing rate above %.0f%% threshold.",
            high_missing,
            threshold * 100,
        )
    return int(high_missing)


def _compute_missing_target_rates(
    splits: dict[SplitName, tuple[pd.DataFrame, pd.DataFrame]],
    targets: tuple[TargetDefinition, ...],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for split_name, (_, target_df) in splits.items():
        for t in targets:
            col = t.column_name
            if col in target_df.columns and len(target_df) > 0:
                rate = float(target_df[col].isna().mean())
                rates[f"{split_name.value}/{col}"] = rate
    return rates


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DatasetBuilderPipeline:
    """
    Joins a FeatureFrame with raw IntervalRecords to produce split datasets.

    This is the final stage of the Phase 2 pipeline:

        ingestion -> feature_engineering -> dataset_builder -> ml/train

    The pipeline outputs one (features, targets) DataFrame pair per split,
    persisted via the injected DatasetStore, alongside a DatasetManifest.

    Features are float32 to reduce training memory cost.
    Targets are float32; binary spike labels are 0.0/1.0/NaN.

    Usage::

        pipeline = DatasetBuilderPipeline(
            config=DatasetBuilderConfig(),
            store=s3_store,
        )
        manifest = pipeline.run(
            feature_frame=frame,
            raw_records=records,
            dataset_id="isem-v1-20240101",
        )
    """

    config: DatasetBuilderConfig = field(default_factory=DatasetBuilderConfig)
    store: DatasetStore | None = None

    def run(
        self,
        feature_frame: FeatureFrame,
        raw_records: Sequence[IntervalRecord],
        dataset_id: str | None = None,
    ) -> DatasetManifest:
        """
        Build, split, and optionally persist a fully labelled dataset.

        Args:
            feature_frame:  output of FeatureEngineeringPipeline.run().
            raw_records:    same normalised records used for feature
                            engineering; used to extract future labels.
            dataset_id:     optional run identifier; defaults to UTC timestamp.

        Returns:
            DatasetManifest describing the full build.

        Raises:
            ValueError:  if inputs are empty or misaligned.
            RuntimeError: if the build produces zero usable rows.
        """
        raw_records = list(raw_records)
        self._validate_inputs(feature_frame, raw_records)

        dataset_id = dataset_id or datetime.now(timezone.utc).strftime(
            "dataset-%Y%m%dT%H%M%SZ"
        )

        logger.info(
            "Dataset build %s started: %d feature vectors, %d raw records, "
            "%d target definitions, config_hash=%s.",
            dataset_id,
            len(feature_frame.vectors),
            len(raw_records),
            len(self.config.targets),
            self.config.to_hash(),
        )

        # ── Step 1: build DataFrames ────────────────────────────────────────
        feature_df = _build_feature_df(feature_frame)
        raw_df = _build_raw_df(raw_records)

        # ── Step 2: extract targets ─────────────────────────────────────────
        target_df = _extract_targets(feature_df, raw_df, self.config.targets)

        # ── Step 3: drop rows where all targets are null ────────────────────
        feature_df, target_df, excluded_null = _drop_all_targets_null(
            feature_df, target_df
        )

        if feature_df.empty:
            raise RuntimeError(
                "No usable rows remain after dropping all-null-target rows. "
                "Ensure raw_records cover the required forecast horizons."
            )

        # ── Step 4: quality metrics ─────────────────────────────────────────
        warned_features = _count_high_missing_features(
            feature_df, self.config.missing_feature_warn_threshold
        )
        mean_feature_missing = float(feature_df.isna().mean().mean())

        # ── Step 5: temporal split ──────────────────────────────────────────
        feature_splits = self.config.split_policy.apply(feature_df)
        target_splits = self.config.split_policy.apply(target_df)

        splits: dict[SplitName, tuple[pd.DataFrame, pd.DataFrame]] = {
            name: (feature_splits[name], target_splits[name])
            for name in SplitName
        }

        # ── Step 6: persist ─────────────────────────────────────────────────
        if self.store is not None:
            for split_name, (feat, tgt) in splits.items():
                self.store.write_split(dataset_id, split_name, feat, tgt)
                logger.info(
                    "Persisted %s split: %d rows.", split_name.value, len(feat)
                )

        # ── Step 7: manifest ────────────────────────────────────────────────
        missing_rates = _compute_missing_target_rates(splits, self.config.targets)

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            built_at=datetime.now(timezone.utc),
            config_hash=self.config.to_hash(),
            feature_count=len(feature_frame.manifest),
            target_definitions=[t.to_dict() for t in self.config.targets],
            split_sizes={
                name.value: len(feat)
                for name, (feat, _) in splits.items()
            },
            missing_target_rates=missing_rates,
            missing_feature_rate_mean=mean_feature_missing,
            window_start_utc=feature_df.index.min().isoformat(),
            window_end_utc=feature_df.index.max().isoformat(),
            rows_excluded_all_targets_null=excluded_null,
            rows_warned_high_missing_features=warned_features,
        )

        if self.store is not None:
            self.store.write_manifest(dataset_id, manifest)

        logger.info(
            "Dataset build %s complete: total=%d, train=%d, val=%d, test=%d, "
            "excluded=%d, config_hash=%s.",
            dataset_id,
            len(feature_df),
            len(splits[SplitName.TRAIN][0]),
            len(splits[SplitName.VALIDATION][0]),
            len(splits[SplitName.TEST][0]),
            excluded_null,
            manifest.config_hash,
        )

        return manifest

    @staticmethod
    def _validate_inputs(
        feature_frame: FeatureFrame,
        raw_records: list[IntervalRecord],
    ) -> None:
        if not feature_frame.vectors:
            raise ValueError("feature_frame contains no vectors.")
        if not raw_records:
            raise ValueError("raw_records cannot be empty.")

        frame_start = feature_frame.vectors[0].timestamp_utc
        frame_end = feature_frame.vectors[-1].timestamp_utc
        raw_start = raw_records[0].timestamp_utc
        raw_end = raw_records[-1].timestamp_utc

        if frame_start < raw_start or frame_end > raw_end:
            logger.warning(
                "Feature frame window [%s → %s] extends outside raw records "
                "window [%s → %s]. Tail rows will have missing targets.",
                frame_start.isoformat(), frame_end.isoformat(),
                raw_start.isoformat(), raw_end.isoformat(),
            )