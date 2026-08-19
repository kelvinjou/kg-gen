"""Parallel vehicle-location timelines and their deterministic oracle."""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    TemporalFact,
)

UTC = timezone.utc


def transit_network_observations() -> tuple[Observation, ...]:
    """Return interleaved vehicle movements, a return edge, and a delayed report."""
    return (
        _location("train-a-east", "Train A", "East Station", 8, 0, 8, 1),
        _location("bus-c-depot-1", "Bus C", "Depot", 8, 5, 8, 6),
        _location("train-b-south", "Train B", "South Station", 8, 15, 8, 16),
        _location("train-a-central", "Train A", "Central Station", 9, 0, 9, 1),
        _location("bus-c-museum", "Bus C", "Museum", 9, 10, 9, 11),
        _location("train-b-central", "Train B", "Central Station", 9, 30, 9, 31),
        _location("train-a-west", "Train A", "West Station", 10, 0, 10, 1),
        _location("bus-c-airport", "Bus C", "Airport", 10, 30, 10, 31),
        _location("train-b-north", "Train B", "North Station", 11, 0, 11, 1),
        _location("train-a-depot", "Train A", "Depot", 12, 0, 12, 1),
        _location("bus-c-depot-2", "Bus C", "Depot", 12, 30, 12, 31),
        _location("train-b-depot", "Train B", "Depot", 13, 0, 13, 1),
        _location("train-a-central-delayed", "Train A", "Central Station", 9, 0, 14, 0),
    )


def _location(
    fact_id: str,
    vehicle: str,
    place: str,
    valid_hour: int,
    valid_minute: int,
    observed_hour: int,
    observed_minute: int,
) -> Observation:
    """Build one vehicle-location observation on the shared case date."""
    return Observation(
        fact_id=fact_id,
        subject=vehicle,
        relation="located_at",
        object=place,
        valid_from=datetime(2026, 9, 2, valid_hour, valid_minute, tzinfo=UTC),
        observed_at=datetime(2026, 9, 2, observed_hour, observed_minute, tzinfo=UTC),
        source_text=f"{vehicle} was reported at {place}.",
    )


class TransitNetworkOracleClassifier:
    """Model each vehicle's changing location edge as an independent timeline."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Distinguish repeated reports, vehicle movement, and shared stations."""
        same_event = old_fact.valid_from == new_fact.valid_from
        if old_fact.triple == new_fact.triple and same_event:
            relationship = "duplicate"
            transition_time = None
        elif (
            old_fact.subject == new_fact.subject
            and old_fact.relation == new_fact.relation
        ):
            relationship = "state_transition"
            transition_time = max(old_fact.valid_from, new_fact.valid_from)
        else:
            relationship = "coexists"
            transition_time = None

        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=relationship,
            transition_time=transition_time,
            rationale="Expected label from the parallel transit-network oracle.",
            decided_at=new_fact.observed_at,
        )
