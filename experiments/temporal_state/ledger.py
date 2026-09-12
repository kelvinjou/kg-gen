"""Append-only storage and runtime reconciliation for temporal facts."""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, get_args

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from experiments.temporal_state.models import (
    ContradictionDimension,
    EvidenceResolution,
    IssueType,
    LifecycleEventType,
    LifecycleResolution,
    LifecycleTransition,
    ReconciliationDecision,
    Relationship,
    TemporalFact,
    utc_now,
)

_RELATIONSHIPS: tuple[Relationship, ...] = get_args(Relationship)


class RelationshipResolutionInstruction(BaseModel):
    """One expert-authored resolution instruction for a known relationship."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution: str = Field(min_length=1)


class LifecycleResolutionInstruction(BaseModel):
    """Natural-language handling instruction for a lifecycle event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution: str = Field(min_length=1)


class ResolutionPolicy(BaseModel):
    """Minimal plug-in contract for expert-authored resolution guidance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    instructions: dict[Relationship, RelationshipResolutionInstruction]
    lifecycle_instructions: dict[
        LifecycleEventType, LifecycleResolutionInstruction
    ] = Field(default_factory=dict)

    @property
    def policy_relationships(self) -> tuple[Relationship, ...]:
        """Return explicit instruction keys, or every relationship by default."""
        return tuple(self.instructions) or _RELATIONSHIPS


def load_resolution_policies(
    policy_path: str | Path,
) -> tuple[ResolutionPolicy, ...]:
    """Load one policy or a bundled policy list and require unique versions."""
    path = Path(policy_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "policies" in payload:
        if set(payload) != {"policies"}:
            raise ValueError("a policy bundle may only contain the policies field")
        policy_payloads = payload["policies"]
        if not isinstance(policy_payloads, list) or not policy_payloads:
            raise ValueError("policies must be a nonempty list")
    else:
        policy_payloads = [payload]

    policies = tuple(
        ResolutionPolicy.model_validate(policy_payload)
        for policy_payload in policy_payloads
    )
    versions = [policy.version for policy in policies]
    if len(set(versions)) != len(versions):
        raise ValueError("policy versions must be unique within a policy bundle")
    return policies


def load_resolution_policy(
    policy_path: str | Path,
    *,
    policy_version: str | None = None,
) -> ResolutionPolicy:
    """Load one policy, selecting its version when the file contains a bundle."""
    policies = load_resolution_policies(policy_path)
    if policy_version is None:
        if len(policies) != 1:
            raise ValueError("policy_version is required for a bundled policy file")
        return policies[0]

    matching = [policy for policy in policies if policy.version == policy_version]
    if not matching:
        raise ValueError(f"unknown policy version: {policy_version}")
    return matching[0]


class RelationshipClassifier(Protocol):
    """Interface for pairwise runtime supersession classification."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify how a newly observed fact affects an existing fact."""
        ...


class LifecyclePolicyResolver(Protocol):
    """Resolve the graph consequence of a unary lifecycle event."""

    def resolve_expiration(self, fact: TemporalFact) -> LifecycleResolution:
        """Apply the configured expiration instruction to one due fact."""
        ...


class _ClassificationOutput(BaseModel):
    """Structured semantic output before ledger identifiers are attached."""

    relationship: Relationship
    issue_type: IssueType | None = None
    contradiction_dimensions: tuple[ContradictionDimension, ...] = ()
    transition_time: datetime | None = None
    rationale: str = ""


class OpenAIRelationshipClassifier:
    """Use an OpenAI-compatible LM Studio model for semantic reconciliation."""

    _ENDPOINT = "http://128.111.28.83:1234/v1"
    _INSTRUCTIONS = """You reconcile two temporal knowledge-graph facts.
Answer the central question: does the NEW fact change whether the OLD relation
should remain visible in a current-state graph?

Use exactly one relationship:
- duplicate: one logical fact expressed again, possibly by another source;
- coexists: both facts can remain visible together;
- state_transition: both were true in sequence and the earlier fact should expire;
- correction: the new assertion says the old assertion was erroneous;
- contradiction: the claims conflict and neither can safely eliminate the other;
- uncertain: the evidence is insufficient to choose another label.

Use the facts' effective times and evidence, not predicate similarity alone.
Facts that were both historically true can still be a state_transition when the
later fact means the earlier relation should no longer appear in the current graph.
For example, started_in at 13:30 followed by extinguished_in at 17:30 is a
state_transition at 17:30. By contrast, affected_air_quality_in and
extinguished_in can coexist because extinguishing a fire does not erase its air
quality effect from the state dimensions represented by those relations.
For state_transition, return the effective transition time with a UTC offset.
For every other label, transition_time must be null.

Also distinguish the semantic issue independently of the legacy relationship:
- logical_contradiction: mutually incompatible in the same scope and valid time;
- temporal_change: different values are valid in sequence;
- granularity_mismatch: population, outcome, dose, geography, or aggregation differs;
- uncertainty: evidence is too incomplete to decide;
- compatible, duplicate, or correction as appropriate.
Record any applicable contradiction_dimensions from source_reliability,
source_recency, temporal_validity, relation_cardinality, negation,
numerical_value, source_provenance, and domain_policy. Different objects alone
are not a contradiction.
"""

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
        use_policy: bool = False,
        policy_path: str | Path | None = None,
        policy_version: str | None = None,
    ) -> None:
        """Configure classification and optional post-classification resolution."""
        load_dotenv()
        self._use_policy = use_policy
        if use_policy and policy_path is None:
            raise ValueError("policy_path is required when use_policy is enabled")
        self._policy = (
            load_resolution_policy(policy_path, policy_version=policy_version)
            if use_policy and policy_path
            else None
        )
        self._policy_relationships = (
            self._policy.policy_relationships if self._policy is not None else ()
        )
        if client is None:
            client = self._create_lmstudio_client()
        self._client = client
        self._model = model or os.environ["LMSTUDIO_MODEL"]

    @property
    def use_policy(self) -> bool:
        """Return whether post-classification evidence handling is active."""
        return self._use_policy

    @property
    def policy(self) -> ResolutionPolicy | None:
        """Return the active evidence-resolution policy, if configured."""
        return self._policy

    @property
    def policy_relationships(self) -> tuple[Relationship, ...]:
        """Return relationships that trigger post-classification policy handling."""
        return self._policy_relationships

    def _policy_system_instructions(self, relationship: Relationship) -> str:
        """Build a resolution prompt that cannot revise the prior classification."""
        if self._policy is None:
            raise RuntimeError("cannot build policy instructions without a policy")

        selected_policy: dict[str, Any] = {
            "name": self._policy.name,
            "version": self._policy.version,
            "domain": self._policy.domain,
        }
        instruction = self._policy.instructions.get(relationship)
        selected_policy["instruction"] = (
            instruction.model_dump(mode="json") if instruction is not None else {}
        )
        policy_json = json.dumps(selected_policy, indent=2)
        return (
            "You resolve nuances between two pieces of evidence after a separate "
            "epistemic classifier has finished.\n"
            + f"The immutable epistemic relationship is {relationship}. Do not "
            + "classify, reclassify, confirm, reject, or return an epistemic "
            + "relationship. Do not alter the classifier's transition time or "
            + "rationale. Apply the relationship-specific instruction below when one "
            + "is provided; otherwise use the default evidence-resolution behavior. "
            + "Return one explicit action: accept_old, accept_new, retain_both, "
            + "defer, or qualify. For accept_old/accept_new, selected_evidence must "
            + "match the action: selected_evidence='old' for accept_old and "
            + "selected_evidence='new' for accept_new. For retain_both/defer it may "
            + "be null. qualify must "
            + "include qualified_assertion and may name provisional selected evidence. "
            + "No record is deleted. Return the action plus one "
            + "resolution narrative in the expert's own domain-appropriate structure; "
            + "do not impose a generic checklist, levels, or categories.\n"
            + policy_json
        )

    def _request_classification(
        self,
        payload: dict[str, Any],
        *,
        system_instructions: str,
    ) -> _ClassificationOutput:
        """Request and validate one structured classification from the model."""
        response = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            response_format=_ClassificationOutput,
            temperature=0.0,
        )
        output = response.choices[0].message.parsed
        if output is None:
            raise RuntimeError("relationship classifier returned no structured output")
        return output

    def _request_policy_resolution(
        self,
        payload: dict[str, Any],
        *,
        relationship: Relationship,
    ) -> EvidenceResolution:
        """Resolve evidence-handling nuance without revising the relationship."""
        response = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": self._policy_system_instructions(relationship),
                },
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            response_format=EvidenceResolution,
            temperature=0.0,
        )
        resolution = response.choices[0].message.parsed
        if resolution is None:
            raise RuntimeError("policy resolver returned no structured output")
        return resolution

    def resolve_expiration(self, fact: TemporalFact) -> LifecycleResolution:
        """Apply the configured natural-language expiration policy."""
        if self._policy is None:
            raise RuntimeError("expiration policy requires use_policy=True")
        instruction = self._policy.lifecycle_instructions.get("expiration")
        if instruction is None:
            raise RuntimeError("policy does not define an expiration instruction")
        selected_policy = {
            "name": self._policy.name,
            "version": self._policy.version,
            "domain": self._policy.domain,
            "event_type": "expiration",
            "instruction": instruction.model_dump(mode="json"),
        }
        response = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You resolve a scheduled knowledge-graph lifecycle event. "
                        "The fact has reached its valid_to boundary and must transition "
                        "from active to expired in the current graph. Apply the supplied "
                        "natural-language policy to explain any review consequence. "
                        "History and provenance must be preserved.\n"
                        + json.dumps(selected_policy, indent=2)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "event_type": "expiration",
                            "effective_at": fact.valid_to,
                            "expiring_fact": fact.model_dump(mode="json"),
                        },
                        indent=2,
                        default=str,
                    ),
                },
            ],
            response_format=LifecycleResolution,
            temperature=0.0,
        )
        resolution = response.choices[0].message.parsed
        if resolution is None:
            raise RuntimeError("lifecycle policy resolver returned no structured output")
        return resolution

    @classmethod
    def _create_lmstudio_client(cls) -> Any:
        """Build an OpenAI client for the sole permitted LM Studio endpoint."""
        endpoint = os.environ["LMSTUDIO_ENDPOINT"]
        if endpoint != cls._ENDPOINT:
            raise ValueError(f"LMSTUDIO_ENDPOINT must be {cls._ENDPOINT}")
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ["LMSTUDIO_API_KEY"],
            base_url=endpoint,
        )

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify once, then optionally resolve evidence-handling nuance."""
        payload = {
            "old_fact": old_fact.model_dump(mode="json"),
            "new_fact": new_fact.model_dump(mode="json"),
        }
        classification = self._request_classification(
            payload,
            system_instructions=self._INSTRUCTIONS,
        )

        policy_applied_for: Relationship | None = None
        policy_resolution: EvidenceResolution | None = None
        if (
            self._policy is not None
            and classification.relationship in self._policy_relationships
        ):
            policy_applied_for = classification.relationship
            print(
                "\tsecond-pass policy resolution: "
                f"{old_fact.fact_id} -> {new_fact.fact_id} "
                f"({policy_applied_for}, policy={self._policy.name})"
            )
            policy_payload = {
                **payload,
                "epistemic_decision": {
                    "relationship": classification.relationship,
                    "transition_time": (
                        classification.transition_time.isoformat()
                        if classification.transition_time is not None
                        else None
                    ),
                    "rationale": classification.rationale,
                },
                "resolution_policy": {
                    "applied_for": policy_applied_for,
                    "name": self._policy.name,
                    "version": self._policy.version,
                },
            }
            policy_resolution = self._request_policy_resolution(
                policy_payload,
                relationship=policy_applied_for,
            )

        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=classification.relationship,
            issue_type=getattr(classification, "issue_type", None),
            contradiction_dimensions=tuple(
                getattr(classification, "contradiction_dimensions", ())
            ),
            transition_time=classification.transition_time,
            rationale=classification.rationale,
            policy_name=self._policy.name if policy_applied_for else None,
            policy_version=self._policy.version if policy_applied_for else None,
            policy_applied_for=policy_applied_for,
            policy_resolution=policy_resolution,
            decided_at=new_fact.observed_at,
        )


class TemporalLedger:
    """Retain assertions and decisions while deriving their effective lifecycle."""

    def __init__(
        self,
        classifier: RelationshipClassifier,
        *,
        max_candidates: int | None = None,
        lifecycle_resolver: LifecyclePolicyResolver | None = None,
    ) -> None:
        """Create an empty ledger with an injected semantic classifier."""
        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self._classifier = classifier
        self._max_candidates = max_candidates
        self._facts: dict[str, TemporalFact] = {}
        self._decisions: list[ReconciliationDecision] = []
        self._lifecycle_resolver = lifecycle_resolver
        classifier_has_policy = hasattr(classifier, "policy")
        classifier_policy = getattr(classifier, "policy", None)
        if (
            self._lifecycle_resolver is None
            and callable(getattr(classifier, "resolve_expiration", None))
            and (
                not classifier_has_policy
                or (
                    classifier_policy is not None
                    and "expiration" in classifier_policy.lifecycle_instructions
                )
            )
        ):
            self._lifecycle_resolver = classifier  # type: ignore[assignment]
        self._expiration_heap: list[tuple[datetime, int, str]] = []
        self._expiration_sequence = 0
        self._lifecycle_time: datetime | None = None
        self._lifecycle_transitions: list[LifecycleTransition] = []
        self._last_lifecycle_transitions: tuple[LifecycleTransition, ...] = ()

    @property
    def facts(self) -> tuple[TemporalFact, ...]:
        """Return raw assertions in ingestion order."""
        return tuple(self._facts.values())

    @property
    def decisions(self) -> tuple[ReconciliationDecision, ...]:
        """Return reconciliation decisions in creation order."""
        return tuple(self._decisions)

    @property
    def lifecycle_transitions(self) -> tuple[LifecycleTransition, ...]:
        """Return scheduled lifecycle transitions in processing order."""
        return tuple(self._lifecycle_transitions)

    @property
    def last_lifecycle_transitions(self) -> tuple[LifecycleTransition, ...]:
        """Return lifecycle transitions drained by the latest ingestion."""
        return self._last_lifecycle_transitions

    def advance_to(
        self, timestamp: datetime, *, processed_at: datetime | None = None
    ) -> tuple[LifecycleTransition, ...]:
        """Drain due expiration events and reconcile state before later work."""
        if timestamp.tzinfo is None:
            raise ValueError("lifecycle timestamp must be timezone-aware")
        if self._lifecycle_time is not None and timestamp < self._lifecycle_time:
            raise ValueError("lifecycle time cannot move backwards")
        self._lifecycle_time = timestamp
        processed_at = processed_at or timestamp
        transitions: list[LifecycleTransition] = []
        while self._expiration_heap and self._expiration_heap[0][0] <= timestamp:
            effective_at, sequence, fact_id = heapq.heappop(self._expiration_heap)
            fact = self._facts.get(fact_id)
            if fact is None or fact.valid_to != effective_at:
                continue
            if any(
                transition.fact_id == fact_id
                and transition.event_type == "expiration"
                and transition.effective_at == effective_at
                for transition in self._lifecycle_transitions
            ):
                continue
            if self._lifecycle_resolver is None:
                resolution = LifecycleResolution(
                    resolution=(
                        "The fact reached valid_to and was removed from the current "
                        "projection while its historical evidence was preserved."
                    )
                )
                policy_name = None
                policy_version = None
            else:
                try:
                    resolution = self._lifecycle_resolver.resolve_expiration(fact)
                except Exception:
                    heapq.heappush(
                        self._expiration_heap, (effective_at, sequence, fact_id)
                    )
                    raise
                policy = getattr(self._lifecycle_resolver, "policy", None)
                policy_name = getattr(policy, "name", None)
                policy_version = getattr(policy, "version", None)
            transition = LifecycleTransition(
                event_type="expiration",
                fact_id=fact_id,
                effective_at=effective_at,
                processed_at=processed_at,
                prior_status="active",
                resulting_status="expired",
                policy_name=policy_name,
                policy_version=policy_version,
                policy_resolution=resolution,
            )
            self._lifecycle_transitions.append(transition)
            transitions.append(transition)
        return tuple(transitions)

    def get_fact(self, fact_id: str) -> TemporalFact:
        """Return a raw assertion by identifier."""
        try:
            return self._facts[fact_id]
        except KeyError as exc:
            raise KeyError(f"unknown fact_id: {fact_id}") from exc

    def retrieve_related_facts(
        self,
        new_fact: TemporalFact,
        *,
        known_at: datetime | None = None,
    ) -> tuple[TemporalFact, ...]:
        """Rank active candidates by exact pair, shared subject, then object."""
        effective = self.effective_facts(known_at=known_at)

        def rank(candidate: TemporalFact) -> int | None:
            """Assign an entity-overlap retrieval rank to one candidate."""
            if (candidate.subject, candidate.object) == (
                new_fact.subject,
                new_fact.object,
            ):
                return 0
            if candidate.subject == new_fact.subject:
                return 1
            if candidate.object == new_fact.object:
                return 2
            return None

        ranked = [
            (candidate_rank, candidate)
            for candidate in effective
            if candidate.fact_id != new_fact.fact_id
            and candidate.status == "active"
            and (candidate_rank := rank(candidate)) is not None
        ]
        ranked.sort(
            key=lambda item: (
                item[0],
                -item[1].valid_from.timestamp(),
                item[1].fact_id,
            )
        )
        candidates = tuple(candidate for _, candidate in ranked)
        if self._max_candidates is not None:
            return candidates[: self._max_candidates]
        return candidates

    def ingest(self, new_fact: TemporalFact) -> tuple[ReconciliationDecision, ...]:
        """Classify a fact against related active facts, then append it."""
        if new_fact.fact_id in self._facts:
            raise ValueError(f"fact_id already exists: {new_fact.fact_id}")

        self._last_lifecycle_transitions = self.advance_to(
            new_fact.observed_at, processed_at=new_fact.observed_at
        )
        decisions: list[ReconciliationDecision] = []
        knowledge_time = max(utc_now(), new_fact.observed_at)
        for old_fact in self.retrieve_related_facts(new_fact, known_at=knowledge_time):
            decision = self._classifier.classify(old_fact, new_fact)
            self._validate_decision(decision, old_fact, new_fact)
            decisions.append(decision)

        self._facts[new_fact.fact_id] = new_fact
        self._decisions.extend(decisions)
        if new_fact.valid_to is not None and new_fact.valid_to > new_fact.observed_at:
            heapq.heappush(
                self._expiration_heap,
                (new_fact.valid_to, self._expiration_sequence, new_fact.fact_id),
            )
            self._expiration_sequence += 1
        return tuple(decisions)

    def ingest_many(
        self, facts: Iterable[TemporalFact]
    ) -> tuple[ReconciliationDecision, ...]:
        """Ingest facts sequentially and return all resulting decisions."""
        return tuple(decision for fact in facts for decision in self.ingest(fact))

    def effective_facts(
        self, *, known_at: datetime | None = None
    ) -> tuple[TemporalFact, ...]:
        """Resolve fact statuses and transition intervals as they were known then."""
        known_at = known_at or utc_now()
        if known_at.tzinfo is None:
            raise ValueError("known_at must be timezone-aware")

        resolved = {
            fact.fact_id: fact
            for fact in self._facts.values()
            if fact.observed_at <= known_at
        }
        decisions = [
            decision
            for decision in self._decisions
            if decision.decided_at <= known_at
            and decision.old_fact_id in resolved
            and decision.new_fact_id in resolved
        ]

        self._apply_corrections(resolved, decisions)
        self._apply_contradictions(resolved, decisions)
        self._apply_transitions(resolved, decisions)
        self._apply_deferred_policy_actions(resolved, decisions)
        self._apply_lifecycle_transitions(resolved, known_at=known_at)
        return tuple(
            resolved[fact_id] for fact_id in self._facts if fact_id in resolved
        )

    def _apply_lifecycle_transitions(
        self, facts: dict[str, TemporalFact], *, known_at: datetime
    ) -> None:
        """Apply processed lifecycle events without changing raw ledger history."""
        for transition in self._lifecycle_transitions:
            if transition.processed_at > known_at or transition.fact_id not in facts:
                continue
            fact = facts[transition.fact_id]
            if fact.status not in {"retracted", "disputed"}:
                facts[transition.fact_id] = fact.model_copy(
                    update={"status": transition.resulting_status}
                )

    @staticmethod
    def _validate_decision(
        decision: ReconciliationDecision,
        old_fact: TemporalFact,
        new_fact: TemporalFact,
    ) -> None:
        """Ensure a classifier decision refers to the supplied fact pair."""
        if decision.old_fact_id != old_fact.fact_id:
            raise ValueError("classifier returned the wrong old_fact_id")
        if decision.new_fact_id != new_fact.fact_id:
            raise ValueError("classifier returned the wrong new_fact_id")

    @staticmethod
    def _apply_corrections(
        facts: dict[str, TemporalFact],
        decisions: list[ReconciliationDecision],
    ) -> None:
        """Mark corrected assertions as retracted without deleting them."""
        for decision in decisions:
            if decision.relationship == "correction":
                old_fact = facts[decision.old_fact_id]
                facts[old_fact.fact_id] = old_fact.model_copy(
                    update={"status": "retracted"}
                )

    @staticmethod
    def _apply_contradictions(
        facts: dict[str, TemporalFact],
        decisions: list[ReconciliationDecision],
    ) -> None:
        """Apply policy selections and retain unselected contradictions as disputed."""
        selected_fact_ids: set[str] = set()
        disputed_fact_ids: set[str] = set()
        for decision in decisions:
            if decision.relationship != "contradiction":
                continue
            resolution = decision.policy_resolution
            if resolution is None:
                disputed_fact_ids.update((decision.old_fact_id, decision.new_fact_id))
                continue

            if resolution.action == "retain_both":
                selected_fact_ids.update((decision.old_fact_id, decision.new_fact_id))
                continue
            if resolution.action == "defer":
                disputed_fact_ids.update((decision.old_fact_id, decision.new_fact_id))
                continue
            if resolution.action == "qualify" and resolution.selected_evidence is None:
                selected_fact_ids.update((decision.old_fact_id, decision.new_fact_id))
                continue

            selected_fact_id = (
                decision.old_fact_id
                if resolution.selected_evidence == "old"
                else decision.new_fact_id
            )
            unselected_fact_id = (
                decision.new_fact_id
                if resolution.selected_evidence == "old"
                else decision.old_fact_id
            )
            selected_fact_ids.add(selected_fact_id)
            disputed_fact_ids.add(unselected_fact_id)

        for fact_id in disputed_fact_ids:
            fact = facts[fact_id]
            if fact.status != "retracted":
                facts[fact_id] = fact.model_copy(update={"status": "disputed"})

        for fact_id in selected_fact_ids - disputed_fact_ids:
            fact = facts[fact_id]
            if fact.status != "retracted":
                facts[fact_id] = fact.model_copy(update={"status": "active"})

    @staticmethod
    def _apply_deferred_policy_actions(
        facts: dict[str, TemporalFact],
        decisions: list[ReconciliationDecision],
    ) -> None:
        """Keep both claims out of the projection when policy explicitly defers."""
        for decision in decisions:
            resolution = decision.policy_resolution
            if resolution is None or resolution.action != "defer":
                continue
            for fact_id in (decision.old_fact_id, decision.new_fact_id):
                fact = facts[fact_id]
                if fact.status != "retracted":
                    facts[fact_id] = fact.model_copy(update={"status": "disputed"})

    @classmethod
    def _apply_transitions(
        cls,
        facts: dict[str, TemporalFact],
        decisions: list[ReconciliationDecision],
    ) -> None:
        """Recompute each classified transition chain by effective time."""
        transition_decisions = [
            decision
            for decision in decisions
            if decision.relationship == "state_transition"
        ]
        linking_decisions = [
            decision
            for decision in decisions
            if decision.relationship in {"state_transition", "correction", "duplicate"}
            and facts[decision.old_fact_id].status != "disputed"
            and facts[decision.new_fact_id].status != "disputed"
        ]
        parent: dict[str, str] = {}

        def find(fact_id: str) -> str:
            """Find a transition component root with path compression."""
            parent.setdefault(fact_id, fact_id)
            if parent[fact_id] != fact_id:
                parent[fact_id] = find(parent[fact_id])
            return parent[fact_id]

        def union(left: str, right: str) -> None:
            """Join two facts into the same transition component."""
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for decision in linking_decisions:
            union(decision.old_fact_id, decision.new_fact_id)

        transition_roots = {
            find(decision.old_fact_id) for decision in transition_decisions
        }
        components: dict[str, list[TemporalFact]] = {}
        for fact_id in parent:
            root = find(fact_id)
            fact = facts[fact_id]
            if root in transition_roots and fact.status != "retracted":
                components.setdefault(root, []).append(fact)

        for component in components.values():
            timeline = sorted(
                component,
                key=lambda fact: (fact.valid_from, fact.observed_at, fact.fact_id),
            )
            groups: list[list[TemporalFact]] = []
            for fact in timeline:
                if not groups or groups[-1][0].valid_from != fact.valid_from:
                    groups.append([])
                groups[-1].append(fact)

            for index, group in enumerate(groups):
                if index == len(groups) - 1:
                    for fact in group:
                        facts[fact.fact_id] = fact.model_copy(
                            update={"status": "active"}
                        )
                    continue

                next_fact = groups[index + 1][0]
                for fact in group:
                    boundary = cls._transition_boundary(
                        fact, next_fact, transition_decisions
                    )
                    if boundary <= fact.valid_from:
                        raise ValueError("transition must occur after the earlier fact")
                    if fact.valid_to is not None:
                        boundary = min(boundary, fact.valid_to)
                    facts[fact.fact_id] = fact.model_copy(
                        update={"valid_to": boundary, "status": "expired"}
                    )

    @staticmethod
    def _transition_boundary(
        earlier: TemporalFact,
        later: TemporalFact,
        decisions: list[ReconciliationDecision],
    ) -> datetime:
        """Choose the recorded boundary for two adjacent transition facts."""
        pair = {earlier.fact_id, later.fact_id}
        matching = [
            decision
            for decision in decisions
            if {decision.old_fact_id, decision.new_fact_id} == pair
            and decision.transition_time is not None
        ]
        if not matching:
            return later.valid_from
        boundary = max(
            matching, key=lambda decision: decision.decided_at
        ).transition_time
        assert boundary is not None
        return boundary
