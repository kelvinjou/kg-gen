"""Behavior checks for the interleaved disaster-mitigation contradiction case."""

import json

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.nonconsecutive_contradictions import (
    NonconsecutiveContradictionOracleClassifier,
    nonconsecutive_contradiction_observations,
)
from experiments.temporal_state.ledger import TemporalLedger
from experiments.temporal_state.project import project_graph
from experiments.temporal_state.visualizer.visualize_kg import visualize_snapshots


def _ingest_case() -> tuple[TemporalLedger, list[tuple]]:
    """Ingest the complete fixture and retain its visualization snapshots."""
    ledger = TemporalLedger(NonconsecutiveContradictionOracleClassifier())
    annotator = TemporalAnnotator()
    snapshots = []
    for observation in nonconsecutive_contradiction_observations():
        decisions = ledger.ingest(annotator.annotate(observation))
        graph = project_graph(
            ledger,
            valid_at=observation.observed_at,
            known_at=observation.observed_at,
        )
        snapshots.append(
            (observation.observed_at.isoformat(), graph, decisions, observation)
        )
    return ledger, snapshots


def test_case_has_twenty_ordered_observations() -> None:
    """Provide exactly 20 distinct reports in observation-time order."""
    observations = nonconsecutive_contradiction_observations()

    assert len(observations) == 20
    assert len({observation.fact_id for observation in observations}) == 20
    assert [observation.observed_at for observation in observations] == sorted(
        observation.observed_at for observation in observations
    )


def test_eight_contradictions_arrive_nonconsecutively() -> None:
    """Space many contradictions across unrelated disaster-response reports."""
    _, snapshots = _ingest_case()
    contradiction_positions = [
        index
        for index, (_, _, decisions, _) in enumerate(snapshots)
        if any(decision.relationship == "contradiction" for decision in decisions)
    ]

    assert contradiction_positions == [3, 6, 9, 11, 13, 15, 17, 19]
    assert all(
        right - left > 1
        for left, right in zip(
            contradiction_positions,
            contradiction_positions[1:],
        )
    )


def test_disputed_claims_do_not_erase_uncontested_mitigation_evidence() -> None:
    """Keep neutral weather, communications, wind, and logistics facts active."""
    ledger, _ = _ingest_case()
    observations = nonconsecutive_contradiction_observations()
    final_time = observations[-1].observed_at
    facts = ledger.effective_facts(known_at=final_time)
    graph = project_graph(ledger, valid_at=final_time, known_at=final_time)

    assert sum(fact.status == "disputed" for fact in facts) == 16
    assert sum(fact.status == "active" for fact in facts) == 4
    assert graph.relations == {
        ("Weather Station Alpha", "reported_condition", "Heavy rain"),
        ("Emergency Radio Network", "communications_status", "Operational"),
        ("Wind Sensor Ridge", "reported_wind", "35 mph northeast"),
        ("Supply Depot West", "staging_status", "Ready"),
    }


def test_visualizer_embeds_all_twenty_raw_observations(tmp_path) -> None:
    """Retain the complete disaster evidence timeline in one HTML artifact."""
    _, snapshots = _ingest_case()
    destination = visualize_snapshots(
        snapshots,
        str(tmp_path / "nonconsecutive-contradictions.html"),
        policy_only_snapshots=True,
    )

    html = destination.read_text(encoding="utf-8")
    data_marker = '<script id="viz-data" type="application/json">\n'
    embedded_json = html.split(data_marker, 1)[1].split("</script>", 1)[0]
    payload = json.loads(embedded_json)

    assert len(payload["snapshots"]) == 20
    assert len(payload["evidence_timeline"]) == 20
    assert payload["timeline_config"] == {
        "policy_only_snapshots": True,
        "enabled_policy_relationships": ["contradiction"],
    }
    assert [
        index
        for index, snapshot in enumerate(payload["snapshots"])
        if snapshot["policy_relevant"]
    ] == [3, 6, 9, 11, 13, 15, 17, 19]
    first_policy_edges = {
        (edge["source"], edge["predicate"], edge["target"])
        for edge in payload["snapshots"][3]["data"]["edges"]
    }
    assert (
        "Weather Station Alpha",
        "reported_condition",
        "Heavy rain",
    ) in first_policy_edges
    assert [entry["snapshot_index"] for entry in payload["evidence_timeline"]] == list(
        range(20)
    )
    assert (
        sum(len(snapshot["manual_overrides"]) for snapshot in payload["snapshots"]) == 8
    )
    assert all(
        len(manual_override["options"]) == 2
        for snapshot in payload["snapshots"]
        for manual_override in snapshot["manual_overrides"]
    )
    assert [
        manual_override["snapshot_index"]
        for snapshot in payload["snapshots"]
        for manual_override in snapshot["manual_overrides"]
    ] == [3, 6, 9, 11, 13, 15, 17, 19]
    assert 'id="policyOnlySnapshots"' in html
    assert "Policy events only" in html
    assert 'id="showContradictionOverrideCard"' in html
    assert "Show override card" in html
