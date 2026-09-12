"""Checks for the factorized vaccine contradiction benchmark."""

import importlib
from collections import Counter

from experiments.temporal_state.cases.vaccine_contradiction_benchmark import (
    GOLD_DECISIONS,
    GOLD_LIFECYCLE_TRANSITIONS,
    POLICY_PATH,
    VaccineGoldClassifier,
    evaluate_vaccine_classifier,
    vaccine_observations,
)
from experiments.temporal_state.cases.vaccine_contradiction_benchmark.run import (
    NewestWinsClassifier,
    NoPolicyClassifier,
)
from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.ledger import TemporalLedger, load_resolution_policy
from experiments.temporal_state.models import EvidenceResolution

run_module = importlib.import_module(
    "experiments.temporal_state.cases.vaccine_contradiction_benchmark.run"
)


def test_vaccine_cases_cover_every_requested_dimension() -> None:
    dimensions = {
        dimension
        for decision in GOLD_DECISIONS
        for dimension in decision.contradiction_dimensions
    }
    assert len(vaccine_observations()) == 2 * len(GOLD_DECISIONS) + 3 == 75
    assert dimensions == {
        "source_reliability",
        "source_recency",
        "temporal_validity",
        "relation_cardinality",
        "negation",
        "numerical_value",
        "source_provenance",
        "domain_policy",
    }


def test_vaccine_policy_defines_expiration_lifecycle_guidance() -> None:
    policy = load_resolution_policy(POLICY_PATH)

    instruction = policy.lifecycle_instructions["expiration"].resolution
    assert "valid_to boundary" in instruction
    assert "preserving its historical interval" in instruction
    assert set(policy.instructions) == {
        "coexists",
        "state_transition",
        "contradiction",
        "uncertain",
    }


def test_vaccine_stress_progression_drains_multiple_expirations() -> None:
    observations = vaccine_observations()

    assert len(GOLD_LIFECYCLE_TRANSITIONS) >= 5
    assert list(observations) == sorted(
        observations, key=lambda observation: observation.observed_at
    )
    assert {transition.fact_id for transition in GOLD_LIFECYCLE_TRANSITIONS} <= {
        observation.fact_id
        for observation in observations
        if observation.valid_to is not None
    }


def test_llm_run_writes_each_viewer_from_the_evaluated_execution(
    tmp_path, monkeypatch
) -> None:
    """Generate policy and generic viewers without classifying either run twice."""
    classifiers = []

    class CountingClassifier:
        def __init__(self, **_kwargs) -> None:
            self.delegate = VaccineGoldClassifier()
            self.calls = 0
            classifiers.append(self)

        def classify(self, old_fact, new_fact):
            self.calls += 1
            return self.delegate.classify(old_fact, new_fact)

    viewer_calls = []

    def record_viewer(
        snapshots,
        destination,
        *,
        include_policy,
        evaluation_reports,
        current_system,
    ):
        viewer_calls.append(
            (
                snapshots,
                destination,
                include_policy,
                evaluation_reports,
                current_system,
            )
        )
        return destination

    monkeypatch.setattr(run_module, "OpenAIRelationshipClassifier", CountingClassifier)
    monkeypatch.setattr(run_module, "write_viewer", record_viewer)
    monkeypatch.setattr(run_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        run_module,
        "POLICY_LLM_VIEWER_PATH",
        tmp_path / "vaccine-contradictions-policy-llm.html",
    )
    monkeypatch.setattr(
        run_module,
        "GENERIC_LLM_VIEWER_PATH",
        tmp_path / "vaccine-contradictions-no-policy.html",
    )

    reports = run_module.run("llm")

    assert set(reports) == {"policy_llm", "generic_llm"}
    assert [classifier.calls for classifier in classifiers] == [36, 36]
    assert [len(call[0]) for call in viewer_calls] == [75, 75]
    assert [call[4] for call in viewer_calls] == ["policy_llm", "generic_llm"]
    assert all(call[3] is reports for call in viewer_calls)
    assert [(call[1].name, call[2]) for call in viewer_calls] == [
        ("vaccine-contradictions-policy-llm.html", True),
        ("vaccine-contradictions-no-policy.html", False),
    ]


def test_gold_labels_do_not_pool_semantically_different_cases() -> None:
    issue_counts = Counter(decision.issue_type for decision in GOLD_DECISIONS)

    assert issue_counts == {
        "logical_contradiction": 8,
        "temporal_change": 7,
        "compatible": 4,
        "granularity_mismatch": 5,
        "uncertainty": 4,
        "duplicate": 4,
        "correction": 4,
    }
    assert {decision.expected_action for decision in GOLD_DECISIONS} == {
        "accept_new",
        "retain_both",
        "qualify",
        "defer",
    }


def test_policy_oracle_scores_each_layer_and_preserves_audit_history() -> None:
    report = evaluate_vaccine_classifier(VaccineGoldClassifier())

    assert report.detection_precision == 1.0
    assert report.detection_recall == 1.0
    assert report.issue_type_accuracy == 1.0
    assert report.dimension_micro_precision == 1.0
    assert report.dimension_micro_recall == 1.0
    assert report.resolution_accuracy == 1.0
    assert report.policy_violation_rate == 0.0
    assert report.audit_preservation_rate == 1.0
    assert report.lifecycle_cases == 11
    assert report.lifecycle_resolution_accuracy == 1.0


def test_newest_wins_can_select_right_claim_but_fail_semantic_diagnosis() -> None:
    report = evaluate_vaccine_classifier(NewestWinsClassifier())

    assert report.issue_type_accuracy < 0.5
    assert report.detection_precision < 1.0
    assert report.resolution_accuracy < 1.0
    assert report.policy_violation_rate > 0.0


def test_no_policy_baseline_does_not_get_resolution_credit_for_abstaining() -> None:
    report = evaluate_vaccine_classifier(NoPolicyClassifier())

    assert report.resolution_accuracy == 0.0
    assert report.policy_violation_rate == 1.0


def test_resolution_actions_are_not_forced_to_choose_old_or_new() -> None:
    retain = EvidenceResolution(action="retain_both", resolution="Both are valid.")
    defer = EvidenceResolution(action="defer", resolution="Review is required.")
    qualify = EvidenceResolution(
        action="qualify",
        qualified_assertion="The estimates apply to different endpoints.",
        resolution="Attach scope before comparison.",
    )

    assert retain.selected_evidence is None
    assert defer.selected_evidence is None
    assert qualify.selected_evidence is None


def test_defer_hides_both_operational_claims_but_retains_raw_evidence() -> None:
    ledger = TemporalLedger(VaccineGoldClassifier())
    ledger.ingest_many(TemporalAnnotator().annotate_many(vaccine_observations()))
    effective = {fact.fact_id: fact for fact in ledger.effective_facts()}

    assert effective["vax-uncertainty-old"].status == "disputed"
    assert effective["vax-uncertainty-new"].status == "disputed"
    assert {fact.fact_id for fact in ledger.facts} >= {
        "vax-uncertainty-old",
        "vax-uncertainty-new",
    }
