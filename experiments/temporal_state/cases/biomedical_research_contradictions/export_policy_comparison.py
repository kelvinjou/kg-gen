"""Maintain a side-by-side CSV of old/new contradiction selections."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.temporal_state.cases.biomedical_research_contradictions import (
    POLICY_PATH,
)

CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output"
OUTPUT_STEM = "biomedical-research-contradictions"
DEFAULT_DESTINATION = OUTPUT_DIR / f"{OUTPUT_STEM}-comparison.csv"

ContradictionKey = tuple[str, str]
PolicyRun = tuple[dict[str, Any], dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _policy_specs(policy_path: Path) -> list[dict[str, Any]]:
    policies = _read_json(policy_path).get("policies")
    if not isinstance(policies, list) or len(policies) != 3:
        raise ValueError(f"Expected exactly three policies in {policy_path}")
    if not all(isinstance(policy, dict) for policy in policies):
        raise ValueError(f"Every policy in {policy_path} must be an object")
    return policies


def _load_available_policy_runs(
    policy_path: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[PolicyRun]]:
    """Ignore absent or stale outputs while the three runs are in progress."""
    policies = _policy_specs(policy_path)
    runs = []
    for policy in policies:
        output_path = output_dir / (
            f"{OUTPUT_STEM}-{policy.get('version', '')}.json"
        )
        if not output_path.is_file():
            continue
        run = _read_json(output_path)
        if run.get("policy_definition") == policy:
            runs.append((policy, run))
    if not runs:
        raise ValueError(
            "No current biomedical policy outputs are available for comparison"
        )
    return policies, runs


def _evidence_by_fact_id(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(observation["fact_id"]): observation
        for item in run.get("evidence_timeline", [])
        if (observation := item.get("observation", {})).get("fact_id")
    }


def _observation_numbers(run: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(observation["fact_id"]): index + 1
        for index, item in enumerate(run.get("evidence_timeline", []))
        if (observation := item.get("observation", {})).get("fact_id")
    }


def _contradiction_selections(
    run: Mapping[str, Any],
) -> dict[ContradictionKey, str]:
    selections = {}
    for snapshot in run.get("snapshots", []):
        for decision in snapshot.get("decisions", []):
            if decision.get("relationship") != "contradiction":
                continue
            key = (
                str(decision.get("old_fact_id", "")),
                str(decision.get("new_fact_id", "")),
            )
            resolution = decision.get("policy_resolution") or {}
            selected = str(resolution.get("selected_evidence", "")).upper()
            selections[key] = (
                selected if selected in {"OLD", "NEW"} else "UNRESOLVED"
            )
    return selections


def _evidence_label(observation: Mapping[str, Any] | None) -> str:
    if not observation:
        return ""
    return (
        f"{observation.get('subject', '')} "
        f"—{observation.get('relation', '')}→ "
        f"{observation.get('object', '')}"
    )


def _policy_column(policy: Mapping[str, Any]) -> str:
    return f"{policy.get('version', '')} {policy.get('name', '')}"


def build_contradiction_rows(
    policies: Sequence[dict[str, Any]],
    runs: Sequence[PolicyRun],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build one row per contradiction and one selection column per policy."""
    policy_columns = [_policy_column(policy) for policy in policies]
    headers = [
        "contradiction_number",
        "observation_number",
        "old_fact_id",
        "old_evidence",
        "new_fact_id",
        "new_evidence",
        "expected",
        *policy_columns,
    ]
    selections_by_version = {
        str(policy["version"]): _contradiction_selections(run)
        for policy, run in runs
    }
    evidence = {}
    observation_numbers = {}
    for _, run in runs:
        evidence.update(_evidence_by_fact_id(run))
        observation_numbers.update(_observation_numbers(run))
    contradiction_keys = sorted(
        {
            key
            for selections in selections_by_version.values()
            for key in selections
        },
        key=lambda key: (
            observation_numbers.get(key[1], float("inf")),
            key[1],
            key[0],
        ),
    )

    rows = []
    completed_versions = set(selections_by_version)
    for index, key in enumerate(contradiction_keys, start=1):
        old_fact_id, new_fact_id = key
        row: dict[str, Any] = {
            "contradiction_number": index,
            "observation_number": observation_numbers.get(new_fact_id, ""),
            "old_fact_id": old_fact_id,
            "old_evidence": _evidence_label(evidence.get(old_fact_id)),
            "new_fact_id": new_fact_id,
            "new_evidence": _evidence_label(evidence.get(new_fact_id)),
            "expected": "",
        }
        for policy in policies:
            version = str(policy["version"])
            if version not in completed_versions:
                result = "NOT RUN"
            else:
                result = selections_by_version[version].get(
                    key, "NO CONTRADICTION"
                )
            row[_policy_column(policy)] = result
        rows.append(row)
    return headers, rows


def update_policy_comparison_csv(
    destination: Path = DEFAULT_DESTINATION,
    *,
    policy_path: Path = POLICY_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Refresh the old/new selection CSV after an individual policy run."""
    destination = destination.resolve()
    expected_by_key: dict[ContradictionKey, str] = {}
    if destination.is_file():
        with destination.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    str(row.get("old_fact_id", "")),
                    str(row.get("new_fact_id", "")),
                )
                expected_by_key[key] = str(row.get("expected", ""))

    policies, runs = _load_available_policy_runs(policy_path, output_dir)
    headers, rows = build_contradiction_rows(policies, runs)
    for row in rows:
        key = (str(row["old_fact_id"]), str(row["new_fact_id"]))
        row["expected"] = expected_by_key.get(key, "")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return destination
