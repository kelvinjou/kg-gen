"""Controlled vaccine claim-conflict benchmark."""

from pathlib import Path

from .scenario import (
    GOLD_DECISIONS,
    GOLD_LIFECYCLE_TRANSITIONS,
    VaccineGoldClassifier,
    evaluate_vaccine_classifier,
    vaccine_observations,
)

POLICY_PATH = Path(__file__).with_name("policy.json")

__all__ = [
    "GOLD_DECISIONS",
    "GOLD_LIFECYCLE_TRANSITIONS",
    "POLICY_PATH",
    "VaccineGoldClassifier",
    "evaluate_vaccine_classifier",
    "vaccine_observations",
]
