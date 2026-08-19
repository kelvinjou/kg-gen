"""Behavior checks for the deliberately noisy harbor-spill case."""

from datetime import datetime, timezone

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.noisy_harbor import (
    INCIDENT,
    NoisyHarborOracleClassifier,
    noisy_harbor_observations,
)
from experiments.temporal_state.ledger import TemporalLedger
from experiments.temporal_state.project import project_graph

UTC = timezone.utc


def _ledger() -> TemporalLedger:
    """Build a fully ingested noisy harbor ledger."""
    ledger = TemporalLedger(NoisyHarborOracleClassifier())
    facts = TemporalAnnotator().annotate_many(noisy_harbor_observations())
    ledger.ingest_many(facts)
    return ledger


def test_noisy_harbor_exercises_non_happy_path_relationships() -> None:
    """Cover contradiction, correction, and uncertainty in one fixture."""
    ledger = _ledger()

    assert {decision.relationship for decision in ledger.decisions} >= {
        "contradiction",
        "correction",
        "uncertain",
    }

    facts = {
        fact.fact_id: fact
        for fact in ledger.effective_facts(
            known_at=datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
        )
    }
    assert facts["location-pier-7"].status == "disputed"
    assert facts["location-pier-9"].status == "disputed"
    assert facts["volume-5000"].status == "retracted"
    assert facts["volume-500-correction"].status == "active"
    assert facts["boom-a-deployed"].status == "active"
    assert facts["boom-alpha-recovered"].status == "active"


def test_noisy_harbor_projection_suppresses_disputed_and_corrected_claims() -> None:
    """Project only the correction and both unresolved uncertain assertions."""
    graph = project_graph(
        _ledger(),
        valid_at=datetime(2026, 9, 4, 10, 0, tzinfo=UTC),
        known_at=datetime(2026, 9, 4, 10, 0, tzinfo=UTC),
    )

    assert graph.relations == {
        (INCIDENT, "estimated_release", "500 gallons"),
        ("Containment Boom A", "deployed_at", "East Channel"),
        ("Boom Alpha", "recovered_from", "East Channel"),
    }


def test_noisy_harbor_reconstructs_pre_correction_knowledge() -> None:
    """Show the initial estimate before its correction became known."""
    graph = project_graph(
        _ledger(),
        valid_at=datetime(2026, 9, 4, 9, 20, tzinfo=UTC),
        known_at=datetime(2026, 9, 4, 9, 20, tzinfo=UTC),
    )

    assert graph.relations == {(INCIDENT, "estimated_release", "5,000 gallons")}
