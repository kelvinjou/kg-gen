"""Run and visualize the 20-observation contradiction case with LM Studio."""

from __future__ import annotations

import re
from pathlib import Path

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.nonconsecutive_contradictions import (
    POLICY_PATH,
    nonconsecutive_contradiction_observations,
)
from experiments.temporal_state.ledger import (
    OpenAIRelationshipClassifier,
    ResolutionPolicy,
    TemporalLedger,
    load_resolution_policies,
    load_resolution_policy,
)
from experiments.temporal_state.project import project_graph
from experiments.temporal_state.visualizer.visualize_kg import (
    write_policy_comparison_dashboard,
    write_snapshot_data,
    write_snapshot_viewer,
)

USE_POLICY = True
POLICY_ONLY_SNAPSHOTS = False
OPEN_IN_BROWSER = True
CASE_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_STEM = "nonconsecutive-contradictions"
VIEWER_PATH = CASE_OUTPUT_DIR / f"{OUTPUT_STEM}.html"
COMPARISON_DASHBOARD_PATH = CASE_OUTPUT_DIR / f"{OUTPUT_STEM}-dashboard.html"


def output_path_for_policy(policy: ResolutionPolicy) -> Path:
    """Build a case-local JSON path ending with the policy version."""
    if re.fullmatch(r"[A-Za-z0-9._-]+", policy.version) is None:
        raise ValueError("policy version contains characters unsafe for a filename")
    return CASE_OUTPUT_DIR / f"{OUTPUT_STEM}-{policy.version}.json"


def run_policy(policy_path: str | Path, policy_version: str) -> Path:
    """Ingest the scenario and write its versioned timeline data as JSON."""
    policy = load_resolution_policy(policy_path, policy_version=policy_version)
    print(f"policy={policy.name} version={policy.version}")
    annotator = TemporalAnnotator()
    ledger = TemporalLedger(
        OpenAIRelationshipClassifier(
            use_policy=USE_POLICY,
            policy_path=policy_path,
            policy_version=policy.version,
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

    destination = write_snapshot_data(
        snapshots,
        output_path_for_policy(policy),
        policy_only_snapshots=POLICY_ONLY_SNAPSHOTS,
        policy=policy,
    )
    print(f"output={destination}")
    return destination


def run() -> tuple[Path, ...]:
    """Generate versioned JSON datasets and their shared HTML viewer."""
    policies = load_resolution_policies(POLICY_PATH)
    data_paths = tuple(
        run_policy(POLICY_PATH, policy.version) for policy in policies
    )
    viewer_path = write_snapshot_viewer(
        {
            policy.version: data_path
            for policy, data_path in zip(policies, data_paths, strict=True)
        },
        VIEWER_PATH,
        default_version=policies[0].version,
        open_in_browser=OPEN_IN_BROWSER,
    )
    dashboard_path = write_policy_comparison_dashboard(COMPARISON_DASHBOARD_PATH)
    print(f"viewer={viewer_path}")
    print(f"dashboard={dashboard_path}")
    return (viewer_path, dashboard_path, *data_paths)


if __name__ == "__main__":
    run()
    # .venv/bin/python -m experiments.temporal_state.cases.nonconsecutive_contradictions.run
