"""Run deterministic baselines or the LM classifier on the vaccine benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.vaccine_contradiction_benchmark import (
    GOLD_DECISIONS,
    GOLD_LIFECYCLE_TRANSITIONS,
    POLICY_PATH,
    VaccineGoldClassifier,
    vaccine_observations,
)
from experiments.temporal_state.evaluation import EvaluationReport, evaluate_decisions
from experiments.temporal_state.ledger import (
    OpenAIRelationshipClassifier,
    RelationshipClassifier,
    TemporalLedger,
    load_resolution_policy,
)
from experiments.temporal_state.models import (
    EvidenceResolution,
    ReconciliationDecision,
    TemporalFact,
)
from experiments.temporal_state.project import project_graph
from experiments.temporal_state.visualizer.visualize_kg import visualize_snapshots

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
VIEWER_PATH = OUTPUT_DIR / "vaccine-contradictions.html"
POLICY_LLM_VIEWER_PATH = OUTPUT_DIR / "vaccine-contradictions-policy-llm.html"
GENERIC_LLM_VIEWER_PATH = OUTPUT_DIR / "vaccine-contradictions-no-policy.html"


class NoPolicyClassifier:
    """Abstaining baseline with no domain policy and no forced winner."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship="uncertain",
            issue_type="uncertainty",
            rationale="No domain policy or relation semantics were supplied.",
            decided_at=new_fact.observed_at,
        )


class NewestWinsClassifier:
    """Intentionally weak baseline that treats every different object as conflict."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship="contradiction",
            issue_type="logical_contradiction",
            contradiction_dimensions=("source_recency",),
            rationale="Baseline selects the most recently observed claim.",
            policy_name="newest_wins",
            policy_version="1.0.0",
            policy_applied_for="contradiction",
            policy_resolution=EvidenceResolution(
                action="accept_new",
                resolution="The new claim was observed later.",
            ),
            decided_at=new_fact.observed_at,
        )


def evaluate_with_snapshots(
    classifier: RelationshipClassifier,
) -> tuple[EvaluationReport, list[tuple[object, ...]]]:
    """Evaluate once while retaining every graph state for visualization."""
    ledger = TemporalLedger(classifier)
    annotator = TemporalAnnotator()
    snapshots = []
    for observation in vaccine_observations():
        decisions = ledger.ingest(annotator.annotate(observation))
        graph = project_graph(
            ledger,
            valid_at=observation.valid_from or observation.observed_at,
            known_at=observation.observed_at,
        )
        snapshots.append(
            (
                observation.observed_at.isoformat(),
                graph,
                decisions,
                observation,
                ledger.last_lifecycle_transitions,
            )
        )
    report = evaluate_decisions(
        GOLD_DECISIONS,
        ledger.decisions,
        retained_fact_ids=(fact.fact_id for fact in ledger.facts),
        gold_lifecycle_transitions=GOLD_LIFECYCLE_TRANSITIONS,
        lifecycle_transitions=ledger.lifecycle_transitions,
    )
    return report, snapshots


def write_viewer(
    snapshots: list[tuple[object, ...]],
    destination: Path,
    *,
    include_policy: bool,
    evaluation_reports: dict[str, dict[str, object]] | None = None,
    current_system: str | None = None,
) -> Path:
    """Write a self-contained step-by-step viewer from an evaluated execution."""
    return visualize_snapshots(
        snapshots,
        str(destination),
        policy=load_resolution_policy(POLICY_PATH) if include_policy else None,
        evaluation_reports=evaluation_reports,
        current_system=current_system,
    )


def run(system: str = "all") -> dict[str, dict[str, object]]:
    """Evaluate selected systems and write a machine-readable score report."""
    classifiers = {}
    if system in {"all", "no-policy"}:
        classifiers["no_policy"] = NoPolicyClassifier()
    if system in {"all", "newest"}:
        classifiers["newest_wins"] = NewestWinsClassifier()
    if system in {"all", "gold"}:
        classifiers["policy_oracle"] = VaccineGoldClassifier()
    if system in {"llm", "llm-policy"}:
        classifiers["policy_llm"] = OpenAIRelationshipClassifier(
            use_policy=True,
            policy_path=POLICY_PATH,
        )
    if system in {"llm", "llm-no-policy"}:
        classifiers["generic_llm"] = OpenAIRelationshipClassifier(use_policy=False)
    reports = {}
    snapshot_sets = {}
    for name, classifier in classifiers.items():
        report, snapshots = evaluate_with_snapshots(classifier)
        reports[name] = report.model_dump(mode="json")
        snapshot_sets[name] = snapshots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "vaccine-contradiction-metrics.json"
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    viewer_specs = {
        "policy_oracle": (VIEWER_PATH, True),
        "policy_llm": (POLICY_LLM_VIEWER_PATH, True),
        "generic_llm": (GENERIC_LLM_VIEWER_PATH, False),
    }
    for name, (viewer_path, include_policy) in viewer_specs.items():
        if name not in snapshot_sets:
            continue
        viewer = write_viewer(
            snapshot_sets[name],
            viewer_path,
            include_policy=include_policy,
            evaluation_reports=reports,
            current_system=name,
        )
        print(f"viewer[{name}]={viewer}")
    print(json.dumps(reports, indent=2))
    print(f"output={destination}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=(
            "all",
            "no-policy",
            "newest",
            "gold",
            "llm",
            "llm-no-policy",
            "llm-policy",
        ),
        default="all",
    )
    args = parser.parse_args()
    run(args.system)


if __name__ == "__main__":
    main()
