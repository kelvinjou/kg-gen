"""Run and visualize the 60-observation biomedical evidence case with LM Studio."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.biomedical_research_contradictions import (
    POLICY_PATH,
    biomedical_evidence_observations,
)
from experiments.temporal_state.cases.biomedical_research_contradictions.export_policy_comparison import (
    update_policy_comparison_csv,
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
OUTPUT_STEM = "biomedical-research-contradictions"
VIEWER_PATH = CASE_OUTPUT_DIR / f"{OUTPUT_STEM}.html"
COMPARISON_DASHBOARD_PATH = CASE_OUTPUT_DIR / f"{OUTPUT_STEM}-dashboard.html"


def snapshot_known_at(
    ledger: TemporalLedger,
    observation_time: datetime,
) -> datetime:
    """Include every reconciliation decision made before projecting a snapshot."""
    return max(
        (
            observation_time,
            *(decision.decided_at for decision in ledger.decisions),
        )
    )


def output_path_for_policy(policy: ResolutionPolicy) -> Path:
    """Build a case-local JSON path ending with the policy version."""
    if re.fullmatch(r"[A-Za-z0-9._-]+", policy.version) is None:
        raise ValueError("policy version contains characters unsafe for a filename")
    return CASE_OUTPUT_DIR / f"{OUTPUT_STEM}-{policy.version}.json"


def run_policy(policy_path: str | Path, policy_version: str) -> Path:
    """Ingest the biomedical scenario and write versioned timeline JSON."""
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

    for observation in biomedical_evidence_observations():
        fact = annotator.annotate(observation)
        decisions = ledger.ingest(fact)
        graph = project_graph(
            ledger,
            valid_at=observation.observed_at,
            known_at=snapshot_known_at(ledger, observation.observed_at),
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
    comparison_path = update_policy_comparison_csv(
        policy_path=Path(policy_path),
        output_dir=CASE_OUTPUT_DIR,
    )
    print(f"comparison_csv={comparison_path}")
    return destination


def run() -> tuple[Path, ...]:
    """Generate three policy datasets, a shared viewer, and comparison dashboard."""
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


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    """Run every policy by default, or rerun one explicitly selected version."""
    parser = argparse.ArgumentParser(
        description="Run the biomedical contradiction-resolution scenario."
    )
    parser.add_argument(
        "--policy-version",
        choices=[policy.version for policy in load_resolution_policies(POLICY_PATH)],
        help="Rerun only this policy version instead of the complete bundle.",
    )
    args = parser.parse_args(argv)
    if args.policy_version is not None:
        return (run_policy(POLICY_PATH, args.policy_version),)
    return run()


if __name__ == "__main__":
    main()
    # .venv/bin/python -m experiments.temporal_state.cases.biomedical_research_contradictions.run
