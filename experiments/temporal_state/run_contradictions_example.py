"""Run and visualize the 20-observation contradiction case with LM Studio."""

from __future__ import annotations

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.nonconsecutive_contradictions import (
    nonconsecutive_contradiction_observations,
)
from experiments.temporal_state.ledger import (
    ENABLED_POLICY_RELATIONSHIPS,
    OpenAIRelationshipClassifier,
    TemporalLedger,
)
from experiments.temporal_state.project import project_graph
from experiments.temporal_state.visualizer.visualize_kg import visualize_snapshots

USE_POLICY = True
POLICY_ONLY_SNAPSHOTS = False


def run() -> None:
    """Ingest all observations and retain each one in a single HTML timeline."""
    annotator = TemporalAnnotator()
    ledger = TemporalLedger(
        OpenAIRelationshipClassifier(
            use_policy=USE_POLICY,
            policy_relationships=ENABLED_POLICY_RELATIONSHIPS,
        )
    )
    snapshots = []

    for observation in nonconsecutive_contradiction_observations():
        fact = annotator.annotate(observation)
        decisions = ledger.ingest(fact)
        graph = project_graph(
            ledger,
            valid_at=observation.observed_at,
            known_at=observation.observed_at,
        )
        snapshots.append(
            (observation.observed_at.isoformat(), graph, decisions, observation)
        )

        labels = ", ".join(decision.relationship for decision in decisions) or "none"
        print(f"{fact.fact_id}: decisions={labels}")

    visualize_snapshots(
        snapshots,
        "experiments/temporal_state/output/nonconsecutive-contradictions.html",
        open_in_browser=True,
        policy_only_snapshots=POLICY_ONLY_SNAPSHOTS,
        enabled_policy_relationships=ENABLED_POLICY_RELATIONSHIPS,
    )


if __name__ == "__main__":
    run()
    # .venv/bin/python -m experiments.temporal_state.run_contradictions_example
