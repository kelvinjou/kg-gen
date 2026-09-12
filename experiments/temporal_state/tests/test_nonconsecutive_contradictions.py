"""Behavior checks for the interleaved disaster-mitigation contradiction case."""

import json
from datetime import timedelta
from types import SimpleNamespace

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.nonconsecutive_contradictions import (
    NonconsecutiveContradictionOracleClassifier,
    POLICY_PATH,
    nonconsecutive_contradiction_observations,
)
from experiments.temporal_state.cases.nonconsecutive_contradictions.run import (
    output_path_for_policy,
    snapshot_known_at,
)
from experiments.temporal_state.ledger import (
    TemporalLedger,
    load_resolution_policies,
    load_resolution_policy,
)
from experiments.temporal_state.project import project_graph
from experiments.temporal_state.visualizer.visualize_kg import (
    visualize_snapshots,
    write_snapshot_data,
    write_snapshot_viewer,
)


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
            known_at=snapshot_known_at(ledger, observation.observed_at),
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


def test_snapshot_knowledge_time_includes_reconciliation_decisions() -> None:
    """Do not hide runtime decisions behind the scenario's timestamps."""
    observation_time = nonconsecutive_contradiction_observations()[-1].observed_at
    decision_time = observation_time + timedelta(days=1)
    ledger = SimpleNamespace(decisions=(SimpleNamespace(decided_at=decision_time),))

    known_at = snapshot_known_at(ledger, observation_time)

    assert all(decision.decided_at <= known_at for decision in ledger.decisions)
    assert known_at == decision_time


def test_case_keeps_its_domain_policy_colocated() -> None:
    """Pair the scenario fixture and its expert policy in one case package."""
    assert POLICY_PATH.name == "policy.json"
    assert POLICY_PATH.parent.name == "nonconsecutive_contradictions"
    assert POLICY_PATH.is_file()


def test_case_policy_variants_have_unique_versions() -> None:
    """Keep every comparison policy valid and independently versioned."""
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policies = load_resolution_policies(POLICY_PATH)

    assert set(payload) == {"policies"}
    assert isinstance(payload["policies"], list)
    assert len(payload["policies"]) == 4
    assert len(policies) == 4
    assert len({policy.version for policy in policies}) == len(policies)
    assert {policy.version for policy in policies} == {
        "5.0.0",
        "5.1.0",
        "5.2.0",
        "5.3.0",
    }
    baseline = next(policy for policy in policies if policy.version == "5.0.0")
    specialized = [policy for policy in policies if policy.version != "5.0.0"]
    assert baseline.instructions == {}
    assert baseline.policy_relationships == (
        "duplicate",
        "coexists",
        "state_transition",
        "correction",
        "contradiction",
        "uncertain",
    )
    assert all(set(policy.instructions) == {"contradiction"} for policy in specialized)


def test_policy_outputs_are_versioned_inside_the_case() -> None:
    """Write each policy's raw timeline to a versioned JSON file."""
    for policy in load_resolution_policies(POLICY_PATH):
        output_path = output_path_for_policy(policy)

        assert output_path.parent == POLICY_PATH.parent / "output"
        assert output_path.name == (
            f"nonconsecutive-contradictions-{policy.version}.json"
        )


def test_shared_viewer_loads_a_versioned_json_timeline(tmp_path) -> None:
    """Keep raw snapshot state outside the one reusable HTML viewer."""
    _, snapshots = _ingest_case()
    policy = load_resolution_policy(POLICY_PATH, policy_version="5.1.0")
    data_path = write_snapshot_data(
        snapshots,
        tmp_path / "nonconsecutive-contradictions-5.1.0.json",
        policy=policy,
    )
    viewer_path = write_snapshot_viewer(
        {policy.version: data_path},
        tmp_path / "nonconsecutive-contradictions.html",
        default_version=policy.version,
    )

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    html = viewer_path.read_text(encoding="utf-8")
    data_marker = '<script id="viz-data" type="application/json">\n'
    embedded_data = html.split(data_marker, 1)[1].split("</script>", 1)[0]
    config_marker = '<script id="viz-data-config" type="application/json">\n'
    embedded_config = html.split(config_marker, 1)[1].split("</script>", 1)[0]

    assert len(payload["snapshots"]) == 20
    assert len(payload["evidence_timeline"]) == 20
    assert payload["policy_definition"] == policy.model_dump(mode="json")
    assert json.loads(embedded_data) == {}
    assert json.loads(embedded_config) == {
        "datasets": [
            {
                "version": "5.1.0",
                "path": "nonconsecutive-contradictions-5.1.0.json",
            }
        ],
        "default_version": "5.1.0",
    }
    assert 'id="datasetVersion"' in html
    assert "fetch(selectedDataset.path" in html
    assert "searchParams.set('version'" in html


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
    known_at = snapshot_known_at(ledger, final_time)
    facts = ledger.effective_facts(known_at=known_at)
    graph = project_graph(ledger, valid_at=final_time, known_at=known_at)

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
        policy=load_resolution_policy(POLICY_PATH, policy_version="5.1.0"),
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
        "available_issue_types": ["logical_contradiction", "compatible"],
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
