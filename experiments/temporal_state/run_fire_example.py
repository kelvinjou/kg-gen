"""Run the transit-network example with LM Studio reconciliation."""

from __future__ import annotations

from kg_gen.kg_gen import KGGen

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.transit_network import transit_network_observations
from experiments.temporal_state.ledger import (
    OpenAIRelationshipClassifier,
    TemporalLedger,
)
from experiments.temporal_state.project import project_graph


def run() -> None:
    """Ingest transit reports and print the graph known at each arrival time."""
    annotator = TemporalAnnotator()
    ledger = TemporalLedger(OpenAIRelationshipClassifier())

    for observation in transit_network_observations():
        fact = annotator.annotate(observation)
        decisions = ledger.ingest(fact)
        graph = project_graph(
            ledger,
            valid_at=observation.observed_at,
            known_at=observation.observed_at,
        )

        KGGen.visualize(graph, "/Volumes/WD1TB/kg-gen/experiments/temporal_state/output/0.html", open_in_browser=True)

        labels = ", ".join(decision.relationship for decision in decisions) or "none"
        print(f"{fact.fact_id}: decisions={labels}")
        for relation in sorted(graph.relations):
            print(f"  {relation}")


if __name__ == "__main__":
    run()
