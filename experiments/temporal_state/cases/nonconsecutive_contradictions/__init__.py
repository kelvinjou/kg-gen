"""Nonconsecutive disaster contradictions and their resolution policy."""

from pathlib import Path

from .scenario import (
    INCIDENT,
    START,
    NonconsecutiveContradictionOracleClassifier,
    nonconsecutive_contradiction_observations,
)

POLICY_PATH = Path(__file__).with_name("policy.json")

__all__ = [
    "INCIDENT",
    "POLICY_PATH",
    "START",
    "NonconsecutiveContradictionOracleClassifier",
    "nonconsecutive_contradiction_observations",
]
