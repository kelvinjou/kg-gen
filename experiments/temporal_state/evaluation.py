"""Independent gold-label evaluation for semantic and policy decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from experiments.temporal_state.models import (
    ContradictionDimension,
    IssueType,
    LifecycleAction,
    LifecycleTransition,
    ReconciliationDecision,
    ResolutionAction,
)


class GoldLifecycleTransition(BaseModel):
    """Expected policy outcome for one scheduled lifecycle event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    action: LifecycleAction
    preserve_history: bool
    review_required: bool


class GoldDecision(BaseModel):
    """Expert-authored expected result for one controlled claim pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    old_fact_id: str
    new_fact_id: str
    issue_type: IssueType
    contradiction_dimensions: tuple[ContradictionDimension, ...]
    expected_action: ResolutionAction
    rationale: str = Field(min_length=1)

    @property
    def pair(self) -> tuple[str, str]:
        return self.old_fact_id, self.new_fact_id


class DimensionScore(BaseModel):
    """Per-dimension exact counts and recall."""

    model_config = ConfigDict(frozen=True)

    gold: int
    predicted: int
    correct: int
    precision: float
    recall: float


class EvaluationReport(BaseModel):
    """Metrics that prevent final-answer accuracy from hiding failure modes."""

    model_config = ConfigDict(frozen=True)

    cases: int
    detection_precision: float
    detection_recall: float
    issue_type_accuracy: float
    dimension_micro_precision: float
    dimension_micro_recall: float
    resolution_accuracy: float
    policy_violation_rate: float
    audit_preservation_rate: float | None
    lifecycle_cases: int = 0
    lifecycle_resolution_accuracy: float | None = None
    by_dimension: dict[ContradictionDimension, DimensionScore]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_decisions(
    gold_decisions: Iterable[GoldDecision],
    predictions: Iterable[ReconciliationDecision],
    *,
    retained_fact_ids: Iterable[str] | None = None,
    gold_lifecycle_transitions: Iterable[GoldLifecycleTransition] = (),
    lifecycle_transitions: Iterable[LifecycleTransition] = (),
) -> EvaluationReport:
    """Score detection, diagnosis, action, policy adherence, and preservation."""
    gold = tuple(gold_decisions)
    predicted_by_pair = {
        (decision.old_fact_id, decision.new_fact_id): decision
        for decision in predictions
    }
    if not gold:
        raise ValueError("gold_decisions cannot be empty")

    logical_gold = {
        item.pair for item in gold if item.issue_type == "logical_contradiction"
    }
    logical_predicted = {
        pair
        for pair, decision in predicted_by_pair.items()
        if decision.issue_type == "logical_contradiction"
    }
    logical_correct = logical_gold & logical_predicted

    issue_correct = 0
    action_correct = 0
    gold_dimensions: Counter[ContradictionDimension] = Counter()
    predicted_dimensions: Counter[ContradictionDimension] = Counter()
    correct_dimensions: Counter[ContradictionDimension] = Counter()
    for item in gold:
        prediction = predicted_by_pair.get(item.pair)
        item_gold_dimensions = set(item.contradiction_dimensions)
        gold_dimensions.update(item_gold_dimensions)
        if prediction is None:
            continue
        issue_correct += prediction.issue_type == item.issue_type
        item_predicted_dimensions = set(prediction.contradiction_dimensions)
        predicted_dimensions.update(item_predicted_dimensions)
        correct_dimensions.update(item_gold_dimensions & item_predicted_dimensions)
        action = (
            prediction.policy_resolution.action
            if prediction.policy_resolution is not None
            else None
        )
        action_correct += action == item.expected_action

    dimensions = sorted(set(gold_dimensions) | set(predicted_dimensions))
    by_dimension = {
        dimension: DimensionScore(
            gold=gold_dimensions[dimension],
            predicted=predicted_dimensions[dimension],
            correct=correct_dimensions[dimension],
            precision=_ratio(
                correct_dimensions[dimension], predicted_dimensions[dimension]
            ),
            recall=_ratio(correct_dimensions[dimension], gold_dimensions[dimension]),
        )
        for dimension in dimensions
    }
    retained = set(retained_fact_ids) if retained_fact_ids is not None else None
    preserved = (
        sum(
            item.old_fact_id in retained and item.new_fact_id in retained
            for item in gold
        )
        if retained is not None
        else None
    )
    gold_dimension_total = sum(gold_dimensions.values())
    predicted_dimension_total = sum(predicted_dimensions.values())
    correct_dimension_total = sum(correct_dimensions.values())
    resolution_accuracy = _ratio(action_correct, len(gold))
    lifecycle_gold = tuple(gold_lifecycle_transitions)
    lifecycle_by_fact = {
        transition.fact_id: transition for transition in lifecycle_transitions
    }
    lifecycle_correct = sum(
        (
            predicted := lifecycle_by_fact.get(expected.fact_id)
        )
        is not None
        and predicted.policy_resolution.action == expected.action
        and predicted.policy_resolution.preserve_history == expected.preserve_history
        and predicted.policy_resolution.review_required == expected.review_required
        for expected in lifecycle_gold
    )
    return EvaluationReport(
        cases=len(gold),
        detection_precision=_ratio(len(logical_correct), len(logical_predicted)),
        detection_recall=_ratio(len(logical_correct), len(logical_gold)),
        issue_type_accuracy=_ratio(issue_correct, len(gold)),
        dimension_micro_precision=_ratio(
            correct_dimension_total, predicted_dimension_total
        ),
        dimension_micro_recall=_ratio(correct_dimension_total, gold_dimension_total),
        resolution_accuracy=resolution_accuracy,
        policy_violation_rate=1.0 - resolution_accuracy,
        audit_preservation_rate=(
            _ratio(preserved, len(gold)) if preserved is not None else None
        ),
        lifecycle_cases=len(lifecycle_gold),
        lifecycle_resolution_accuracy=(
            _ratio(lifecycle_correct, len(lifecycle_gold))
            if lifecycle_gold
            else None
        ),
        by_dimension=by_dimension,
    )
