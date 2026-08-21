"""Behavior checks for parallel vehicle-location timelines."""

from datetime import datetime, timezone

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.transit_network import (
    TransitNetworkOracleClassifier,
    transit_network_observations,
)
from experiments.temporal_state.ledger import TemporalLedger
from experiments.temporal_state.project import project_graph

UTC = timezone.utc


def _ledger() -> TemporalLedger:
    """Build a fully ingested transit-network ledger."""
    ledger = TemporalLedger(TransitNetworkOracleClassifier())
    facts = TemporalAnnotator().annotate_many(transit_network_observations())
    ledger.ingest_many(facts)
    return ledger


def test_transit_entities_move_independently_to_final_locations() -> None:
    """Keep one final location edge for each moving entity."""
    graph = project_graph(
        _ledger(),
        valid_at=datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
        known_at=datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
    )

    assert graph.relations == {
        ("Train A", "located_at", "Depot"),
        ("Train B", "located_at", "Depot"),
        ("Bus C", "located_at", "Depot"),
    }


def test_transit_projection_reconstructs_parallel_mid_route_locations() -> None:
    """Recover each vehicle's location from a historical snapshot."""
    graph = project_graph(
        _ledger(),
        valid_at=datetime(2026, 9, 2, 9, 15, tzinfo=UTC),
        known_at=datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
    )

    assert graph.relations == {
        ("Train A", "located_at", "Central Station"),
        ("Train B", "located_at", "South Station"),
        ("Bus C", "located_at", "Museum"),
    }
