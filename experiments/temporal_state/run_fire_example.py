"""Run the noisy harbor example with LM Studio reconciliation."""

from __future__ import annotations

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.noisy_harbor import noisy_harbor_observations
from experiments.temporal_state.ledger import (
    OpenAIRelationshipClassifier,
    TemporalLedger,
)
from experiments.temporal_state.project import project_graph
from experiments.temporal_state.visualizer.visualize_kg import visualize_snapshots

# This scenario has no colocated domain policy. Use the dedicated contradiction
# runner when testing policy-driven resolution.
USE_POLICY = False
POLICY_ONLY_SNAPSHOTS = False


def run() -> None:
    """Ingest noisy harbor reports and visualize each observed state."""
    annotator = TemporalAnnotator()
    ledger = TemporalLedger(
        OpenAIRelationshipClassifier(
            use_policy=USE_POLICY,
        )
    )
    snapshots = []

    for observation in noisy_harbor_observations():
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
        for relation in sorted(graph.relations):
            print(f"  {relation}")

    visualize_snapshots(
        snapshots,
        "experiments/temporal_state/output/harbor.html",
        open_in_browser=True,
        policy_only_snapshots=POLICY_ONLY_SNAPSHOTS,
    )


if __name__ == "__main__":
    run()
    # open_output_visualization("harbor.html")
