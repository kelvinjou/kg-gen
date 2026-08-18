"""Behavior checks for temporal-state ingestion and projection."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.cases.fire import (
    FireOracleClassifier,
    fire_observations,
)
from experiments.temporal_state.ledger import (
    OpenAIRelationshipClassifier,
    TemporalLedger,
)
from experiments.temporal_state.models import ReconciliationDecision, TemporalFact
from experiments.temporal_state.project import project_graph

UTC = timezone.utc


def test_openai_classifier_uses_lmstudio_environment(monkeypatch) -> None:
    """Configure the OpenAI-compatible client exclusively from LM Studio values."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", "test-lmstudio-key")
    monkeypatch.setenv("LMSTUDIO_ENDPOINT", "http://128.111.28.83:1234/v1")
    monkeypatch.setenv("LMSTUDIO_MODEL", "test-local-model")
    fake_client = Mock()

    with patch("openai.OpenAI", return_value=fake_client) as constructor:
        OpenAIRelationshipClassifier()

    constructor.assert_called_once_with(
        api_key="test-lmstudio-key",
        base_url="http://128.111.28.83:1234/v1",
    )


def test_delayed_fire_report_does_not_replace_latest_state() -> None:
    """Keep the extinguished state after a late-arriving older report."""
    ledger = TemporalLedger(FireOracleClassifier())
    annotator = TemporalAnnotator()
    ledger.ingest_many(annotator.annotate_many(fire_observations()))

    graph = project_graph(
        ledger,
        valid_at=datetime(2026, 8, 16, 18, 30, tzinfo=UTC),
        known_at=datetime(2026, 8, 16, 18, 30, tzinfo=UTC),
    )

    assert (
        "Fremont Fire 2026-08-16",
        "extinguished_in",
        "Fremont",
    ) in graph.relations
    assert not any(relation == "started_in" for _, relation, _ in graph.relations)
    assert any(
        relation == "affected_air_quality_in" for _, relation, _ in graph.relations
    )


def test_projection_uses_valid_and_observation_times() -> None:
    """Exclude knowledge not yet observed while honoring historical valid time."""
    ledger = TemporalLedger(FireOracleClassifier())
    annotator = TemporalAnnotator()
    ledger.ingest_many(annotator.annotate_many(fire_observations()))

    graph = project_graph(
        ledger,
        valid_at=datetime(2026, 8, 16, 14, 30, tzinfo=UTC),
        known_at=datetime(2026, 8, 16, 14, 30, tzinfo=UTC),
    )

    assert {predicate for _, predicate, _ in graph.relations} == {
        "started_in",
        "affected_air_quality_in",
    }


class _CorrectionClassifier:
    """Classify a replacement as a correction for a focused ledger test."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Return a deterministic correction decision."""
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship="correction",
            confidence=1.0,
            decided_at=new_fact.observed_at,
        )


def test_correction_retracts_old_assertion() -> None:
    """Retain corrected history while omitting it from the outward graph."""
    first = TemporalFact(
        fact_id="wrong",
        subject="Fire",
        relation="extinguished_in",
        object="Fremont",
        valid_from=datetime(2026, 8, 16, 17, 30, tzinfo=UTC),
        observed_at=datetime(2026, 8, 16, 17, 31, tzinfo=UTC),
        source_text="The fire was extinguished.",
    )
    correction = TemporalFact(
        fact_id="corrected",
        subject="Fire",
        relation="contained_in",
        object="Fremont",
        valid_from=datetime(2026, 8, 16, 17, 30, tzinfo=UTC),
        observed_at=datetime(2026, 8, 16, 18, 0, tzinfo=UTC),
        source_text="Correction: the fire was contained, not extinguished.",
    )
    ledger = TemporalLedger(_CorrectionClassifier())
    ledger.ingest_many((first, correction))

    effective = {fact.fact_id: fact for fact in ledger.effective_facts()}
    assert effective["wrong"].status == "retracted"
    assert effective["corrected"].status == "active"
    assert len(ledger.facts) == 2


class _TransitionCorrectionClassifier:
    """Classify one transition followed by a correction to its latest fact."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Return the expected decision for the test's ordered inputs."""
        if new_fact.relation == "extinguished_in":
            relationship = "state_transition"
            transition_time = new_fact.valid_from
        else:
            relationship = "correction"
            transition_time = None
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=relationship,
            transition_time=transition_time,
            confidence=1.0,
            decided_at=new_fact.observed_at,
        )


def test_correction_preserves_the_transition_chain() -> None:
    """Prevent an earlier state from resurfacing after correcting its successor."""
    facts = (
        TemporalFact(
            fact_id="started",
            subject="Fire",
            relation="started_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 13, 30, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 13, 31, tzinfo=UTC),
            source_text="The fire started.",
        ),
        TemporalFact(
            fact_id="extinguished",
            subject="Fire",
            relation="extinguished_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 17, 30, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 17, 31, tzinfo=UTC),
            source_text="The fire was extinguished.",
        ),
        TemporalFact(
            fact_id="contained",
            subject="Fire",
            relation="contained_in",
            object="Fremont",
            valid_from=datetime(2026, 8, 16, 17, 30, tzinfo=UTC),
            observed_at=datetime(2026, 8, 16, 18, 0, tzinfo=UTC),
            source_text="Correction: contained, not extinguished.",
        ),
    )
    ledger = TemporalLedger(_TransitionCorrectionClassifier())
    ledger.ingest_many(facts)

    graph = project_graph(
        ledger,
        valid_at=datetime(2026, 8, 16, 18, 30, tzinfo=UTC),
        known_at=datetime(2026, 8, 16, 18, 30, tzinfo=UTC),
    )

    assert graph.relations == {("Fire", "contained_in", "Fremont")}
