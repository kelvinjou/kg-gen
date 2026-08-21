"""Checks for post-classification evidence-resolution policies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.temporal_state.ledger import (
    DEFAULT_POLICY_PATH,
    ENABLED_POLICY_RELATIONSHIPS,
    OpenAIRelationshipClassifier,
    load_resolution_policy,
)
from experiments.temporal_state.models import EvidenceResolution, TemporalFact

UTC = timezone.utc


def _fact(fact_id: str, object_: str, minute: int) -> TemporalFact:
    """Build one fact for focused classifier tests."""
    timestamp = datetime(2026, 9, 4, 9, minute, tzinfo=UTC)
    return TemporalFact(
        fact_id=fact_id,
        subject="Bayview Harbor Spill",
        relation="reported_source_at",
        object=object_,
        valid_from=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
        observed_at=timestamp,
        source_text=f"A report placed the source at {object_}.",
    )


def _client_with_result(relationship: str = "contradiction") -> Mock:
    """Return a mock client with separate classification and resolution results."""
    client = Mock()
    classification = SimpleNamespace(
        relationship=relationship,
        transition_time=(
            datetime(2026, 9, 4, 9, 10, tzinfo=UTC)
            if relationship == "state_transition"
            else None
        ),
        rationale="The assertions conflict for the same effective interval.",
    )
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=classification))]
        )
    ]
    if relationship == "contradiction":
        resolution = EvidenceResolution(
            resolution=(
                "Both reports are direct but neither is independently corroborated. "
                "Preserve both and use a precautionary envelope pending a geolocated "
                "field observation."
            ),
        )
        responses.append(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=resolution))]
            )
        )
    client.chat.completions.parse.side_effect = responses
    return client


def test_policy_contains_resolution_guidance_not_classification_rules() -> None:
    """Keep policy.json downstream from the epistemic classification contract."""
    policy = load_resolution_policy()

    assert DEFAULT_POLICY_PATH.name == "policy.json"
    assert policy.domain == "Emergency and disaster mitigation"
    assert set(policy.instructions) == {"contradiction"}
    assert "reversible response posture" in (
        policy.instructions["contradiction"].resolution
    )
    serialized_policy = policy.model_dump_json()
    assert "proportionate, reversible response posture" in serialized_policy
    for classification_key in (
        "epistemic_vocabulary",
        '"definition"',
        "choose_when",
        "avoid_when",
        "fallback_relationship",
        "graph_effect",
        "resolution_protocol",
        "interim_mitigation",
        "resolution_evidence",
        "output_requirements",
    ):
        assert classification_key not in serialized_policy
    for scenario_term in ("Pier 7", "Pier 9", "Coast Guard", "harbor", "spill", "Boom"):
        assert scenario_term not in serialized_policy


def test_domain_expert_policy_is_plug_and_play(tmp_path: Path) -> None:
    """Accept another expert's guidance without changing code or policy shape."""
    policy_path = tmp_path / "public-health-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "name": "public_health_evidence_resolution",
                "version": "1.0.0",
                "domain": "Public health incident management",
                "instructions": {
                    "contradiction": {
                        "resolution": (
                            "Interpret unresolved evidence using epidemiological "
                            "source quality, sampling bias, and the cost of delayed "
                            "intervention."
                        )
                    },
                    "correction": {
                        "resolution": "Use the public-health correction protocol."
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    client = _client_with_result()
    classifier = OpenAIRelationshipClassifier(
        model="test-model",
        client=client,
        use_policy=True,
        policy_path=policy_path,
    )

    decision = classifier.classify(
        _fact("report-a", "Zone A", 3),
        _fact("report-b", "Zone B", 8),
    )

    policy_prompt = client.chat.completions.parse.call_args_list[1].kwargs["messages"][
        0
    ]["content"]
    assert "epidemiological source quality" in policy_prompt
    assert "public-health correction protocol" not in policy_prompt
    assert decision.relationship == "contradiction"
    assert decision.policy_name == "public_health_evidence_resolution"


def test_contradiction_triggers_resolution_without_reclassification() -> None:
    """Keep the first LLM decision immutable while resolving evidence nuance."""
    client = _client_with_result()
    classifier = OpenAIRelationshipClassifier(
        model="test-model",
        client=client,
        use_policy=True,
    )

    decision = classifier.classify(
        _fact("pier-7", "Pier 7", 3),
        _fact("pier-9", "Pier 9", 8),
    )

    assert ENABLED_POLICY_RELATIONSHIPS == ["contradiction"]
    assert client.chat.completions.parse.call_count == 2
    initial_request, policy_request = (
        call.kwargs for call in client.chat.completions.parse.call_args_list
    )
    assert initial_request["response_format"].__name__ == "_ClassificationOutput"
    assert policy_request["response_format"] is EvidenceResolution
    system_prompt = policy_request["messages"][0]["content"]
    user_payload = json.loads(policy_request["messages"][1]["content"])
    assert "immutable epistemic relationship is contradiction" in system_prompt
    assert "Do not classify, reclassify" in system_prompt
    assert "Use exactly one relationship" not in system_prompt
    assert "duplicate:" not in system_prompt
    assert '"definition"' not in system_prompt
    assert '"choose_when"' not in system_prompt
    assert '"instruction"' in system_prompt
    assert "reversible response posture" in system_prompt
    assert "expert's own domain-appropriate structure" in system_prompt
    assert user_payload["epistemic_decision"]["relationship"] == "contradiction"
    assert user_payload["resolution_policy"] == {
        "applied_for": "contradiction",
        "name": "disaster_mitigation_evidence_resolution",
        "version": "4.0.0",
    }
    assert decision.relationship == "contradiction"
    assert decision.rationale == (
        "The assertions conflict for the same effective interval."
    )
    assert decision.policy_name == "disaster_mitigation_evidence_resolution"
    assert decision.policy_version == "4.0.0"
    assert decision.policy_applied_for == "contradiction"
    assert decision.policy_resolution is not None
    assert "Preserve both and use a precautionary envelope" in (
        decision.policy_resolution.resolution
    )


@pytest.mark.parametrize(
    "relationship",
    ["duplicate", "coexists", "state_transition", "correction", "uncertain"],
)
def test_non_enabled_relationships_do_not_apply_policy(relationship: str) -> None:
    """Keep every non-enabled epistemic relationship on the base prompt only."""
    client = _client_with_result(relationship)
    classifier = OpenAIRelationshipClassifier(
        model="test-model",
        client=client,
        use_policy=True,
    )

    decision = classifier.classify(
        _fact("pier-7", "Pier 7", 3),
        _fact("pier-9", "Pier 9", 8),
    )

    request = client.chat.completions.parse.call_args.kwargs
    system_prompt = request["messages"][0]["content"]
    user_payload = json.loads(request["messages"][1]["content"])
    assert client.chat.completions.parse.call_count == 1
    assert "immutable epistemic relationship" not in system_prompt
    assert "disaster_mitigation_evidence_resolution" not in system_prompt
    assert "policy" not in user_payload
    assert "resolution_policy" not in user_payload
    assert decision.relationship == relationship
    assert decision.policy_name is None
    assert decision.policy_version is None
    assert decision.policy_applied_for is None
    assert decision.policy_resolution is None


def test_policy_is_opt_in() -> None:
    """Preserve the original prompt and decision metadata when the flag is false."""
    client = _client_with_result("uncertain")
    classifier = OpenAIRelationshipClassifier(
        model="test-model",
        client=client,
        use_policy=False,
    )

    decision = classifier.classify(
        _fact("pier-7", "Pier 7", 3),
        _fact("pier-9", "Pier 9", 8),
    )

    request = client.chat.completions.parse.call_args.kwargs
    assert "immutable epistemic relationship" not in request["messages"][0]["content"]
    assert "policy" not in json.loads(request["messages"][1]["content"])
    assert decision.policy_name is None
    assert decision.policy_version is None
    assert decision.policy_applied_for is None
    assert decision.policy_resolution is None
