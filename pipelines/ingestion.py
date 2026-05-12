"""
Ingestion pipeline for the Energy Decision Engine.

Orchestrates fetching, validation, normalisation, and storage of
market data from SEMO, EirGrid, and weather sources.

Design principles:
- Each source is fetched independently; partial failure does not abort the run.
- All raw payloads are persisted before transformation (raw-first strategy).
- Schema validation is applied before any downstream write.
- Every ingestion run produces a structured IngestionReport for observability.
- No silent data loss: every rejected record is logged with a reason.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class SourceName(str, Enum):
    SEMO = "semo"
    EIRGRID = "eirgrid"
    WEATHER = "weather"


class IngestionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# Seconds to wait between retry attempts (exponential base).
_RETRY_BASE_DELAY_S: float = 1.0
_MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Raw record and validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawRecord:
    """
    One logical row from an external source before any transformation.

    source:         which feed produced this record
    timestamp_utc:  the market interval this row represents
    payload:        raw key/value pairs exactly as received
    ingested_at:    wall-clock UTC time this record entered the pipeline
    """

    source: SourceName
    timestamp_utc: datetime
    payload: dict[str, Any]
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError(
                f"timestamp_utc must be timezone-aware; got {self.timestamp_utc!r}"
            )


@dataclass(frozen=True)
class ValidationError:
    record: RawRecord
    reason: str


@dataclass
class ValidationResult:
    valid: list[RawRecord] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        total = len(self.valid) + len(self.errors)
        return len(self.errors) / total if total else 0.0


# ---------------------------------------------------------------------------
# Ingestion report (observability contract)
# ---------------------------------------------------------------------------

@dataclass
class SourceReport:
    source: SourceName
    status: IngestionStatus
    records_fetched: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    rejection_rate: float = 0.0
    duration_s: float = 0.0
    error: str | None = None
    retries: int = 0


@dataclass
class IngestionReport:
    """
    Structured summary of a full pipeline run.

    Intended to be serialised (JSON/DynamoDB) for alerting and dashboards.
    """

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    overall_status: IngestionStatus = IngestionStatus.SUCCESS
    source_reports: list[SourceReport] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    def finalise(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        statuses = {r.status for r in self.source_reports}
        if all(s == IngestionStatus.SUCCESS for s in statuses):
            self.overall_status = IngestionStatus.SUCCESS
        elif all(s == IngestionStatus.FAILED for s in statuses):
            self.overall_status = IngestionStatus.FAILED
        else:
            self.overall_status = IngestionStatus.PARTIAL


# ---------------------------------------------------------------------------
# Ports (dependency inversion — implementations live in infrastructure/)
# ---------------------------------------------------------------------------

class RawFetcher(Protocol):
    """Fetch raw records from a single external source for a time window."""

    @property
    def source_name(self) -> SourceName: ...

    def fetch(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> Iterator[RawRecord]: ...


class RecordValidator(Protocol):
    """Validate a batch of raw records for a given source."""

    def validate(
        self,
        source: SourceName,
        records: Sequence[RawRecord],
    ) -> ValidationResult: ...


class RawStore(Protocol):
    """Persist raw records before transformation (raw-first strategy)."""

    def write(self, records: Sequence[RawRecord]) -> None: ...


class NormalisedStore(Protocol):
    """Persist validated, normalised records for downstream consumption."""

    def write(self, source: SourceName, records: Sequence[RawRecord]) -> None: ...


# ---------------------------------------------------------------------------
# Validator base — extend per source in infrastructure/
# ---------------------------------------------------------------------------

class BaseValidator(ABC):
    """
    Shared validation logic.  Source-specific validators subclass this
    and override `_required_fields` and `_validate_record`.
    """

    @property
    @abstractmethod
    def _required_fields(self) -> frozenset[str]: ...

    def validate(
        self,
        source: SourceName,
        records: Sequence[RawRecord],
    ) -> ValidationResult:
        result = ValidationResult()
        for record in records:
            error = self._check(record)
            if error:
                result.errors.append(ValidationError(record=record, reason=error))
                logger.warning(
                    "Rejected record from %s at %s: %s",
                    source.value,
                    record.timestamp_utc.isoformat(),
                    error,
                )
            else:
                result.valid.append(record)
        return result

    def _check(self, record: RawRecord) -> str | None:
        missing = self._required_fields - record.payload.keys()
        if missing:
            return f"missing required fields: {sorted(missing)}"
        return self._validate_record(record)

    @abstractmethod
    def _validate_record(self, record: RawRecord) -> str | None:
        """Return a human-readable error string, or None if valid."""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _fetch_with_retry(
    fetcher: RawFetcher,
    start_utc: datetime,
    end_utc: datetime,
    max_retries: int = _MAX_RETRIES,
    base_delay_s: float = _RETRY_BASE_DELAY_S,
) -> tuple[list[RawRecord], int]:
    """
    Call fetcher.fetch with exponential backoff.

    Returns (records, retries_used).  Raises the last exception if all
    attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            records = list(fetcher.fetch(start_utc, end_utc))
            return records, attempt
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay_s * (2 ** attempt)
                logger.warning(
                    "Fetch attempt %d/%d failed for %s: %s. Retrying in %.1fs.",
                    attempt + 1,
                    max_retries + 1,
                    fetcher.source_name.value,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d fetch attempts failed for %s: %s",
                    max_retries + 1,
                    fetcher.source_name.value,
                    exc,
                )
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Per-source ingestion unit
# ---------------------------------------------------------------------------

def _ingest_source(
    fetcher: RawFetcher,
    validator: RecordValidator,
    raw_store: RawStore,
    normalised_store: NormalisedStore,
    start_utc: datetime,
    end_utc: datetime,
) -> SourceReport:
    """
    Full ingestion cycle for one source:
      1. fetch (with retry)
      2. persist raw (raw-first — never lose the original payload)
      3. validate
      4. persist valid normalised records
    """
    report = SourceReport(source=fetcher.source_name, status=IngestionStatus.FAILED)
    t0 = time.monotonic()

    try:
        logger.info(
            "Starting ingestion for %s [%s → %s]",
            fetcher.source_name.value,
            start_utc.isoformat(),
            end_utc.isoformat(),
        )

        records, retries = _fetch_with_retry(fetcher, start_utc, end_utc)
        report.retries = retries
        report.records_fetched = len(records)

        if not records:
            logger.warning(
                "%s returned 0 records for window [%s → %s].",
                fetcher.source_name.value,
                start_utc.isoformat(),
                end_utc.isoformat(),
            )
            report.status = IngestionStatus.SUCCESS
            return report

        # Raw-first: persist before any transformation so we can reprocess.
        raw_store.write(records)

        validation = validator.validate(fetcher.source_name, records)
        report.records_accepted = len(validation.valid)
        report.records_rejected = len(validation.errors)
        report.rejection_rate = validation.rejection_rate

        if validation.rejection_rate > 0.5:
            logger.error(
                "%s rejection rate %.1f%% exceeds threshold. Aborting normalised write.",
                fetcher.source_name.value,
                validation.rejection_rate * 100,
            )
            report.status = IngestionStatus.PARTIAL
            return report

        if validation.valid:
            normalised_store.write(fetcher.source_name, validation.valid)

        report.status = (
            IngestionStatus.SUCCESS
            if not validation.errors
            else IngestionStatus.PARTIAL
        )

        logger.info(
            "%s ingestion complete: %d accepted, %d rejected (%.1f%%), %d retries.",
            fetcher.source_name.value,
            report.records_accepted,
            report.records_rejected,
            report.rejection_rate * 100,
            report.retries,
        )

    except Exception as exc:  # noqa: BLE001
        report.status = IngestionStatus.FAILED
        report.error = str(exc)
        logger.exception(
            "Ingestion failed for %s: %s",
            fetcher.source_name.value,
            exc,
        )
    finally:
        report.duration_s = time.monotonic() - t0

    return report


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

@dataclass
class IngestionPipeline:
    """
    Orchestrates ingestion across all configured sources.

    Sources are processed independently so that one failure does not
    prevent others from completing.  The resulting IngestionReport
    captures per-source outcomes and an overall status.

    Usage::

        pipeline = IngestionPipeline(
            fetchers=[semo_fetcher, eirgrid_fetcher, weather_fetcher],
            validators={
                SourceName.SEMO: semo_validator,
                SourceName.EIRGRID: eirgrid_validator,
                SourceName.WEATHER: weather_validator,
            },
            raw_store=s3_raw_store,
            normalised_store=dynamodb_store,
        )

        report = pipeline.run(start_utc=..., end_utc=...)
    """

    fetchers: list[RawFetcher]
    validators: dict[SourceName, RecordValidator]
    raw_store: RawStore
    normalised_store: NormalisedStore

    def run(
        self,
        start_utc: datetime,
        end_utc: datetime,
        run_id: str | None = None,
    ) -> IngestionReport:
        """
        Execute the full ingestion pipeline.

        Args:
            start_utc:  inclusive start of the market window to ingest.
            end_utc:    exclusive end of the market window to ingest.
            run_id:     optional caller-supplied identifier; defaults to
                        an ISO timestamp of when the run started.

        Returns:
            IngestionReport with per-source outcomes and overall status.

        Raises:
            ValueError: if start_utc >= end_utc or either is naive.
        """
        self._validate_window(start_utc, end_utc)

        started_at = datetime.now(timezone.utc)
        run_id = run_id or started_at.strftime("run-%Y%m%dT%H%M%SZ")
        report = IngestionReport(run_id=run_id, started_at=started_at)

        logger.info(
            "Ingestion run %s started: window [%s → %s], sources=%s",
            run_id,
            start_utc.isoformat(),
            end_utc.isoformat(),
            [f.source_name.value for f in self.fetchers],
        )

        for fetcher in self.fetchers:
            validator = self.validators.get(fetcher.source_name)
            if validator is None:
                logger.error(
                    "No validator registered for source %s — skipping.",
                    fetcher.source_name.value,
                )
                report.source_reports.append(
                    SourceReport(
                        source=fetcher.source_name,
                        status=IngestionStatus.FAILED,
                        error="no validator configured",
                    )
                )
                continue

            source_report = _ingest_source(
                fetcher=fetcher,
                validator=validator,
                raw_store=self.raw_store,
                normalised_store=self.normalised_store,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            report.source_reports.append(source_report)

        report.finalise()

        logger.info(
            "Ingestion run %s finished in %.2fs — overall status: %s",
            run_id,
            report.duration_s,
            report.overall_status.value,
        )

        return report

    @staticmethod
    def _validate_window(start_utc: datetime, end_utc: datetime) -> None:
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start_utc and end_utc must be timezone-aware.")
        if start_utc >= end_utc:
            raise ValueError(
                f"start_utc must be before end_utc; got {start_utc!r} >= {end_utc!r}"
            )