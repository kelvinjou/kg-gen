"""Structural and temporal checks for the biomedical evidence case."""

import csv
import json
from collections import Counter
from datetime import timedelta
from types import SimpleNamespace

from experiments.temporal_state.cases.biomedical_research_contradictions import (
    EXPECTED_RELATIONSHIP_COUNTS,
    PAIR_RELATIONSHIPS,
    POLICY_PATH,
    biomedical_evidence_observations,
    build_biomedical_evidence_ledger,
    relationship_counts,
)
from experiments.temporal_state.cases.biomedical_research_contradictions.run import (
    COMPARISON_DASHBOARD_PATH,
    OUTPUT_STEM,
    VIEWER_PATH,
    main,
    output_path_for_policy,
    snapshot_known_at,
)
from experiments.temporal_state.cases.biomedical_research_contradictions.export_policy_comparison import (
    OUTPUT_STEM as COMPARISON_OUTPUT_STEM,
    update_policy_comparison_csv,
)
from experiments.temporal_state.ledger import load_resolution_policies
from experiments.temporal_state.project import project_graph


def test_case_has_sixty_bitemporal_observations() -> None:
    observations = biomedical_evidence_observations()

    assert len(observations) == 60
    assert len({observation.fact_id for observation in observations}) == 60
    assert [observation.observed_at for observation in observations] == sorted(
        observation.observed_at for observation in observations
    )
    assert all(
        observation.valid_from < observation.observed_at
        for observation in observations
        if observation.valid_from is not None
    )


def test_case_has_ten_curated_contradiction_pairs() -> None:
    assert len(PAIR_RELATIONSHIPS) == 30
    assert Counter(PAIR_RELATIONSHIPS.values()) == Counter(
        EXPECTED_RELATIONSHIP_COUNTS
    )
    assert Counter(PAIR_RELATIONSHIPS.values())["contradiction"] == 10


def test_ingestion_produces_balanced_relationships() -> None:
    ledger = build_biomedical_evidence_ledger()

    assert len(ledger.facts) == 60
    assert len(ledger.decisions) == 30
    assert relationship_counts(ledger) == EXPECTED_RELATIONSHIP_COUNTS


def test_ten_contradictions_are_delayed_and_nonconsecutive() -> None:
    observations = biomedical_evidence_observations()
    ledger = build_biomedical_evidence_ledger()
    positions = {observation.fact_id: index for index, observation in enumerate(observations)}
    contradiction_positions = [
        positions[decision.new_fact_id]
        for decision in ledger.decisions
        if decision.relationship == "contradiction"
    ]

    assert contradiction_positions == [8, 11, 20, 23, 32, 35, 44, 47, 56, 59]
    assert all(right - left > 1 for left, right in zip(contradiction_positions, contradiction_positions[1:]))
    assert all(
        positions[decision.new_fact_id] - positions[decision.old_fact_id] >= 6
        for decision in ledger.decisions
        if decision.relationship == "contradiction"
    )


def test_projection_has_depth_and_preserves_history() -> None:
    observations = biomedical_evidence_observations()
    ledger = build_biomedical_evidence_ledger()
    graph = project_graph(
        ledger,
        valid_at=observations[-1].observed_at,
        known_at=snapshot_known_at(ledger, observations[-1].observed_at),
    )

    assert len(graph.entities) >= 35
    assert len(graph.edges) >= 12
    assert len(graph.relations) >= 25
    assert "CLARITY AD evidence node" in graph.entities
    assert "TRAILBLAZER-ALZ 2 evidence node" in graph.entities
    assert "Lecanemab regimen node" in graph.entities


def test_snapshot_knowledge_time_includes_reconciliation_decisions() -> None:
    """Do not hide runtime decisions behind the scenario's historical timestamps."""
    observation_time = biomedical_evidence_observations()[-1].observed_at
    decision_time = observation_time + timedelta(days=1)
    ledger = SimpleNamespace(
        decisions=(SimpleNamespace(decided_at=decision_time),)
    )

    known_at = snapshot_known_at(ledger, observation_time)

    assert all(decision.decided_at <= known_at for decision in ledger.decisions)
    assert known_at == decision_time


def test_colocated_policy_has_baseline_domain_and_mismatched_variants() -> None:
    policies = load_resolution_policies(POLICY_PATH)

    assert len(policies) == 3
    assert {policy.version for policy in policies} == {"1.0.0", "1.2.0", "2.0.0"}
    baseline = next(policy for policy in policies if policy.version == "1.0.0")
    specialized = [policy for policy in policies if policy.version != "1.0.0"]
    assert baseline.instructions == {}
    assert all(
        set(policy.instructions) == {"contradiction"}
        for policy in specialized
    )


def test_runner_outputs_are_versioned_inside_the_case() -> None:
    """Keep all runner artifacts colocated and distinguish policy versions."""
    for policy in load_resolution_policies(POLICY_PATH):
        output_path = output_path_for_policy(policy)

        assert output_path.parent == POLICY_PATH.parent / "output"
        assert output_path.name == f"{OUTPUT_STEM}-{policy.version}.json"

    assert VIEWER_PATH == POLICY_PATH.parent / "output" / f"{OUTPUT_STEM}.html"
    assert COMPARISON_DASHBOARD_PATH == (
        POLICY_PATH.parent / "output" / f"{OUTPUT_STEM}-dashboard.html"
    )


def test_comparison_csv_has_one_wide_row_per_observation(tmp_path) -> None:
    policy_payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    output_dir = tmp_path / "policy-outputs"
    output_dir.mkdir()
    observation = {
        "subject": "Study conclusion",
        "relation": "supports",
        "object": "Treatment benefit",
        "source_text": "Synthetic test evidence",
        "observed_at": "2026-01-01T00:00:00Z",
        "valid_from": "2025-01-01T00:00:00Z",
        "valid_to": None,
        "fact_id": "fact-new",
    }

    for index, policy in enumerate(policy_payload["policies"]):
        relation = {
            "source": "Study conclusion",
            "predicate": "supports" if index < 2 else "disputes",
            "target": "Treatment benefit",
        }
        run_payload = {
            "policy_definition": policy,
            "evidence_timeline": [
                {
                    "snapshot_index": 0,
                    "snapshot_label": "2026-01-01T00:00:00Z",
                    "observation": observation,
                }
            ],
            "snapshots": [
                {
                    "data": {"relations": [relation]},
                    "decisions": [
                        {
                            "old_fact_id": "fact-old",
                            "new_fact_id": "fact-new",
                            "relationship": "contradiction",
                            "policy_resolution": {
                                "selected_evidence": (
                                    "old" if index < 2 else "new"
                                ),
                                "resolution": "Synthetic selection",
                            },
                        }
                    ],
                    "changes": {
                        "added": [
                            [
                                relation["source"],
                                relation["predicate"],
                                relation["target"],
                            ]
                        ],
                        "removed": [],
                    },
                }
            ],
        }
        path = output_dir / (
            f"{COMPARISON_OUTPUT_STEM}-{policy['version']}.json"
        )
        path.write_text(json.dumps(run_payload), encoding="utf-8")

    destination = update_policy_comparison_csv(
        tmp_path / "comparison.csv",
        output_dir=output_dir,
    )

    with destination.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["old_fact_id"] == "fact-old"
    assert rows[0]["new_fact_id"] == "fact-new"
    assert rows[0]["expected"] == ""
    assert list(rows[0]).index("expected") < list(rows[0]).index(
        "1.0.0 biomedical_research_baseline"
    )
    assert rows[0]["1.0.0 biomedical_research_baseline"] == "OLD"
    assert rows[0]["1.2.0 biomedical_evidence_quality_adjudication"] == "OLD"
    assert rows[0]["2.0.0 biomedical_regulatory_actionability_adjudication"] == "NEW"

    rows[0]["expected"] = "UNRESOLVED"
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    update_policy_comparison_csv(destination, output_dir=output_dir)

    with destination.open(encoding="utf-8-sig", newline="") as handle:
        refreshed_rows = list(csv.DictReader(handle))
    assert refreshed_rows[0]["expected"] == "UNRESOLVED"


def test_runner_can_select_one_policy_version(monkeypatch) -> None:
    calls = []

    def fake_run_policy(policy_path, policy_version):
        calls.append((policy_path, policy_version))
        return policy_path.parent / "output" / f"selected-{policy_version}.json"

    monkeypatch.setattr(
        "experiments.temporal_state.cases.biomedical_research_contradictions.run.run_policy",
        fake_run_policy,
    )

    outputs = main(["--policy-version", "2.0.0"])

    assert calls == [(POLICY_PATH, "2.0.0")]
    assert outputs == (POLICY_PATH.parent / "output" / "selected-2.0.0.json",)
