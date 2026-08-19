"""Fremont fire inputs and an explicit offline oracle for the demo case."""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    TemporalFact,
)

UTC = timezone.utc


def fire_observations() -> tuple[Observation, ...]:
    """Return independent reports, including one delayed duplicate report."""
    return (
        Observation(
            fact_id="fire-started",
            subject="Fremont Fire 2026-08-16",
            relation="started_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 13, 30, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 13, 31, tzinfo=UTC),
            source_text="At 1:30 PM, the fire in Fremont started.",
        ),
        Observation(
            fact_id="fire-air-quality",
            subject="Fremont Fire 2026-08-16",
            relation="affected_air_quality_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 14, 0, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 14, 5, tzinfo=UTC),
            source_text="The Fremont fire affected local air quality.",
        ),
        Observation(
            fact_id="fire-extinguished",
            subject="Fremont Fire 2026-08-16",
            relation="extinguished_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 17, 30, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 17, 31, tzinfo=UTC),
            source_text="At 5:30 PM, the fire in Fremont was extinguished.",
        ),
        Observation(
            fact_id="fire-started-delayed",
            subject="Fremont Fire 2026-08-16",
            relation="started_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 13, 30, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 18, 0, tzinfo=UTC),
            source_text="A delayed report confirms the fire started at 1:30 PM.",
        ),
    )


class FireOracleClassifier:
    """Provide deterministic expected labels for the offline fire example only."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify the small fixture without pretending to be a general ontology."""
        if old_fact.triple == new_fact.triple:
            relationship = "duplicate"
            transition_time = None
        elif {old_fact.relation, new_fact.relation} == {
            "started_in",
            "extinguished_in",
        }:
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
            rationale="Expected label from the deterministic fire-case oracle.",
            decided_at=new_fact.observed_at,
        )
