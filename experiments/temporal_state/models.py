"""Data models for the temporal-state experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FactStatus = Literal["active", "expired", "retracted", "disputed"]
Relationship = Literal[
    "duplicate",
    "coexists",
    "state_transition",
    "correction",
    "contradiction",
    "uncertain",
]
Relation = tuple[str, str, str]


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Observation(BaseModel):
    """A raw KGGen relation with evidence and temporal metadata."""

    model_config = ConfigDict(frozen=True)

    subject: str
    relation: str
    object: str
    source_text: str
    observed_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    fact_id: str | None = None

    @field_validator("subject", "relation", "object", "source_text")
    @classmethod
    def require_text(cls, value: str) -> str:
        """Trim text fields and reject empty values."""
        value = value.strip()
        if not value:
            raise ValueError("observation text fields cannot be empty")
        return value

    @field_validator("observed_at", "valid_from", "valid_to")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Reject timestamps that do not include a UTC offset."""
        if value is not None and value.tzinfo is None:
            raise ValueError("temporal timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_ordered_interval(self) -> "Observation":
        """Require an optional validity interval to move forward in time."""
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValueError("valid_to must be later than valid_from")
        return self


class TemporalFact(BaseModel):
    """An immutable relation assertion retained in the historical ledger."""

    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str
    relation: str
    object: str
    valid_from: datetime
    valid_to: datetime | None = None
    observed_at: datetime
    source_text: str
    status: FactStatus = "active"

    @field_validator("subject", "relation", "object", "source_text")
    @classmethod
    def require_text(cls, value: str) -> str:
        """Trim text fields and reject empty values."""
        value = value.strip()
        if not value:
            raise ValueError("fact text fields cannot be empty")
        return value

    @field_validator("valid_from", "valid_to", "observed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Reject timestamps that do not include a UTC offset."""
        if value is not None and value.tzinfo is None:
            raise ValueError("temporal timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_ordered_interval(self) -> "TemporalFact":
        """Require a nonempty half-open validity interval."""
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be later than valid_from")
        return self

    @property
    def triple(self) -> Relation:
        """Return the fact as a KGGen-compatible relation tuple."""
        return self.subject, self.relation, self.object

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Return whether the fact's half-open interval contains a time."""
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return self.valid_from <= timestamp and (
            self.valid_to is None or timestamp < self.valid_to
        )


class ReconciliationDecision(BaseModel):
    """An immutable runtime classification between two ledger facts."""

    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    old_fact_id: str
    new_fact_id: str
    relationship: Relationship
    transition_time: datetime | None = None
    rationale: str = ""
    decided_at: datetime = Field(default_factory=utc_now)

    @field_validator("transition_time", "decided_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Reject decision timestamps that omit a UTC offset."""
        if value is not None and value.tzinfo is None:
            raise ValueError("temporal timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_relationship(self) -> "ReconciliationDecision":
        """Validate identifiers and transition-specific fields."""
        if self.old_fact_id == self.new_fact_id:
            raise ValueError("a decision must relate two different facts")
        if self.relationship == "state_transition" and self.transition_time is None:
            raise ValueError("state transitions require transition_time")
        if self.relationship != "state_transition" and self.transition_time is not None:
            raise ValueError("only state transitions may set transition_time")
        return self


class StateProjection(BaseModel):
    """A bitemporal snapshot that can be converted to a KGGen Graph."""

    model_config = ConfigDict(frozen=True)

    valid_at: datetime
    known_at: datetime
    facts: tuple[TemporalFact, ...]
    relations: frozenset[Relation]

    @field_validator("valid_at", "known_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject projection timestamps that omit a UTC offset."""
        if value.tzinfo is None:
            raise ValueError("temporal timestamps must be timezone-aware")
        return value
