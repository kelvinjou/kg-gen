"""Data models for the temporal-state experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FactStatus = Literal["active", "expired", "retracted", "disputed"]
LifecycleEventType = Literal["expiration"]
LifecycleAction = Literal["expire"]
Relationship = Literal[
    "duplicate",
    "coexists",
    "state_transition",
    "correction",
    "contradiction",
    "uncertain",
]
Relation = tuple[str, str, str]
IssueType = Literal[
    "duplicate",
    "compatible",
    "logical_contradiction",
    "temporal_change",
    "granularity_mismatch",
    "uncertainty",
    "correction",
]
ContradictionDimension = Literal[
    "source_reliability",
    "source_recency",
    "temporal_validity",
    "relation_cardinality",
    "negation",
    "numerical_value",
    "source_provenance",
    "domain_policy",
]
ResolutionAction = Literal[
    "accept_old",
    "accept_new",
    "retain_both",
    "defer",
    "qualify",
]


_ISSUE_FOR_RELATIONSHIP: dict[Relationship, IssueType] = {
    "duplicate": "duplicate",
    "coexists": "compatible",
    "state_transition": "temporal_change",
    "correction": "correction",
    "contradiction": "logical_contradiction",
    "uncertain": "uncertainty",
}


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Provenance(BaseModel):
    """Structured source metadata kept separate from the evidence quotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str | None = None
    source_type: str | None = None
    publisher: str | None = None
    uri: str | None = None
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    methodological_quality: float | None = Field(default=None, ge=0.0, le=1.0)


class Observation(BaseModel):
    """A raw KGGen relation with evidence and temporal metadata."""

    model_config = ConfigDict(frozen=True)

    subject: str
    relation: str
    object: str
    source_text: str
    provenance: "Provenance" = Field(default_factory=lambda: Provenance())
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
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
    provenance: "Provenance" = Field(default_factory=lambda: Provenance())
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
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


class EvidenceResolution(BaseModel):
    """Auditable policy action applied after an epistemic decision.

    ``selected_evidence`` is retained for old datasets.  New experiments should
    score ``action`` so abstention and preservation are not forced into a winner.
    """

    model_config = ConfigDict(frozen=True)

    action: ResolutionAction | None = None
    selected_evidence: Literal["old", "new"] | None = None
    qualified_assertion: str | None = None
    review_required: bool = False
    resolution: str = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_legacy_selection(self) -> "EvidenceResolution":
        """Keep old/new callers compatible while making the action explicit."""
        action = self.action
        selected = self.selected_evidence
        if action is None and selected is None:
            raise ValueError("resolution requires an action or selected_evidence")
        if action is None:
            action = "accept_old" if selected == "old" else "accept_new"
            object.__setattr__(self, "action", action)
        expected = {
            "accept_old": "old",
            "accept_new": "new",
        }.get(action)
        if expected is not None:
            if selected is not None and selected != expected:
                raise ValueError("action and selected_evidence disagree")
            object.__setattr__(self, "selected_evidence", expected)
        if action == "qualify" and not self.qualified_assertion:
            raise ValueError("qualify requires qualified_assertion")
        return self


class LifecycleResolution(BaseModel):
    """Validated graph effect returned by a natural-language lifecycle policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: LifecycleAction = "expire"
    preserve_history: bool = True
    review_required: bool = False
    resolution: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_history_preservation(self) -> "LifecycleResolution":
        """Expiration may change current state but never erase its evidence."""
        if not self.preserve_history:
            raise ValueError("lifecycle resolution must preserve history")
        return self


class LifecycleTransition(BaseModel):
    """Auditable unary state transition caused by a scheduled lifecycle event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: LifecycleEventType
    fact_id: str
    effective_at: datetime
    processed_at: datetime
    prior_status: FactStatus
    resulting_status: FactStatus
    policy_name: str | None = None
    policy_version: str | None = None
    policy_resolution: LifecycleResolution

    @field_validator("effective_at", "processed_at")
    @classmethod
    def require_event_timezone(cls, value: datetime) -> datetime:
        """Reject lifecycle timestamps that omit a UTC offset."""
        if value.tzinfo is None:
            raise ValueError("lifecycle timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_expiration(self) -> "LifecycleTransition":
        """Keep expiration transitions unary and semantically consistent."""
        if self.event_type == "expiration" and self.resulting_status != "expired":
            raise ValueError("expiration must result in expired status")
        if (self.policy_name is None) != (self.policy_version is None):
            raise ValueError("policy_name and policy_version must be set together")
        return self


class ReconciliationDecision(BaseModel):
    """An immutable runtime classification between two ledger facts."""

    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    old_fact_id: str
    new_fact_id: str
    relationship: Relationship
    issue_type: IssueType | None = None
    contradiction_dimensions: tuple[ContradictionDimension, ...] = ()
    transition_time: datetime | None = None
    rationale: str = ""
    policy_name: str | None = None
    policy_version: str | None = None
    policy_applied_for: Relationship | None = None
    policy_resolution: EvidenceResolution | None = None
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
        if self.issue_type is None:
            object.__setattr__(
                self, "issue_type", _ISSUE_FOR_RELATIONSHIP[self.relationship]
            )
        if len(set(self.contradiction_dimensions)) != len(
            self.contradiction_dimensions
        ):
            raise ValueError("contradiction_dimensions must be unique")
        policy_metadata = (
            self.policy_name,
            self.policy_version,
            self.policy_applied_for,
            self.policy_resolution,
        )
        if any(value is not None for value in policy_metadata) and not all(
            value is not None for value in policy_metadata
        ):
            raise ValueError(
                "policy_name, policy_version, policy_applied_for, and "
                "policy_resolution must be set together"
            )
        if (
            self.policy_applied_for is not None
            and self.policy_applied_for != self.relationship
        ):
            raise ValueError(
                "policy_applied_for must match the immutable epistemic relationship"
            )
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
