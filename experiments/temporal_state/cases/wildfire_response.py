"""Branching wildfire-response observations and their deterministic oracle."""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    TemporalFact,
)

UTC = timezone.utc
INCIDENT = "Redwood Fire 2026-09-01"


def wildfire_response_observations() -> tuple[Observation, ...]:
    """Return interleaved updates for four independently changing incident branches."""
    return (
        Observation(
            fact_id="north-warning",
            subject=INCIDENT,
            relation="warning_for",
            object="North Zone",
            valid_from=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 8, 2, tzinfo=UTC),
            source_text="An evacuation warning was issued for the North Zone.",
        ),
        Observation(
            fact_id="south-warning",
            subject=INCIDENT,
            relation="warning_for",
            object="South Zone",
            valid_from=datetime(2026, 9, 1, 8, 15, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 8, 18, tzinfo=UTC),
            source_text="An evacuation warning was issued for the South Zone.",
        ),
        Observation(
            fact_id="highway-closed",
            subject=INCIDENT,
            relation="closed_route",
            object="Highway 9",
            valid_from=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 9, 1, tzinfo=UTC),
            source_text="Highway 9 was closed because of the fire.",
        ),
        Observation(
            fact_id="north-order",
            subject=INCIDENT,
            relation="order_for",
            object="North Zone",
            valid_from=datetime(2026, 9, 1, 9, 30, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 9, 32, tzinfo=UTC),
            source_text="The North Zone warning became an evacuation order.",
        ),
        Observation(
            fact_id="shelter-opened",
            subject=INCIDENT,
            relation="opened_shelter_at",
            object="Civic Center",
            valid_from=datetime(2026, 9, 1, 9, 45, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 9, 47, tzinfo=UTC),
            source_text="An evacuation shelter opened at the Civic Center.",
        ),
        Observation(
            fact_id="south-order",
            subject=INCIDENT,
            relation="order_for",
            object="South Zone",
            valid_from=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 10, 2, tzinfo=UTC),
            source_text="The South Zone warning became an evacuation order.",
        ),
        Observation(
            fact_id="north-lifted",
            subject=INCIDENT,
            relation="order_lifted_for",
            object="North Zone",
            valid_from=datetime(2026, 9, 1, 14, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 14, 3, tzinfo=UTC),
            source_text="The evacuation order for the North Zone was lifted.",
        ),
        Observation(
            fact_id="highway-reopened",
            subject=INCIDENT,
            relation="reopened_route",
            object="Highway 9",
            valid_from=datetime(2026, 9, 1, 15, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 15, 4, tzinfo=UTC),
            source_text="Highway 9 reopened to traffic.",
        ),
        Observation(
            fact_id="south-lifted",
            subject=INCIDENT,
            relation="order_lifted_for",
            object="South Zone",
            valid_from=datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 16, 2, tzinfo=UTC),
            source_text="The evacuation order for the South Zone was lifted.",
        ),
        Observation(
            fact_id="shelter-closed",
            subject=INCIDENT,
            relation="closed_shelter_at",
            object="Civic Center",
            valid_from=datetime(2026, 9, 1, 18, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 18, 5, tzinfo=UTC),
            source_text="The Civic Center evacuation shelter closed.",
        ),
        Observation(
            fact_id="south-warning-delayed",
            subject=INCIDENT,
            relation="warning_for",
            object="South Zone",
            valid_from=datetime(2026, 9, 1, 8, 15, tzinfo=UTC),
            observed_at=datetime(2026, 9, 1, 19, 0, tzinfo=UTC),
            source_text="A delayed bulletin confirmed the South Zone warning.",
        ),
    )


class WildfireResponseOracleClassifier:
    """Treat each incident-to-resource pair as an independent state branch."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify same-time repeats, within-branch changes, and parallel branches."""
        same_event = old_fact.valid_from == new_fact.valid_from
        if old_fact.triple == new_fact.triple and same_event:
            relationship = "duplicate"
            transition_time = None
        elif (old_fact.subject, old_fact.object) == (
            new_fact.subject,
            new_fact.object,
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
            rationale="Expected label from the branching wildfire-response oracle.",
            decided_at=new_fact.observed_at,
        )
