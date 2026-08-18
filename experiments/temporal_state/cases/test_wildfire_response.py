"""Behavior checks for the branching wildfire-response case."""

from datetime import datetime, timezone

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.wildfire_response import (
    INCIDENT,
    WildfireResponseOracleClassifier,
    wildfire_response_observations,
)
from experiments.temporal_state.ledger import TemporalLedger
from experiments.temporal_state.project import project_graph

UTC = timezone.utc


def _ledger() -> TemporalLedger:
    """Build a fully ingested wildfire-response ledger."""
    ledger = TemporalLedger(WildfireResponseOracleClassifier())
    facts = TemporalAnnotator().annotate_many(wildfire_response_observations())
    ledger.ingest_many(facts)
    return ledger


def test_wildfire_branches_reach_independent_final_states() -> None:
    """Keep the latest edge on every incident branch."""
    graph = project_graph(
        _ledger(),
        valid_at=datetime(2026, 9, 1, 19, 30, tzinfo=UTC),
        known_at=datetime(2026, 9, 1, 19, 30, tzinfo=UTC),
    )

    assert graph.relations == {
        (INCIDENT, "order_lifted_for", "North Zone"),
        (INCIDENT, "order_lifted_for", "South Zone"),
        (INCIDENT, "reopened_route", "Highway 9"),
        (INCIDENT, "closed_shelter_at", "Civic Center"),
    }


def test_wildfire_projection_reconstructs_mid_incident_branches() -> None:
    """Recover all four branch states from the middle of the incident."""
    graph = project_graph(
        _ledger(),
        valid_at=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        known_at=datetime(2026, 9, 1, 19, 30, tzinfo=UTC),
    )

    assert graph.relations == {
        (INCIDENT, "order_for", "North Zone"),
        (INCIDENT, "order_for", "South Zone"),
        (INCIDENT, "closed_route", "Highway 9"),
        (INCIDENT, "opened_shelter_at", "Civic Center"),
    }
