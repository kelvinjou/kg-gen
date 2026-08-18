"""Append-only storage and runtime reconciliation for temporal facts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from experiments.temporal_state.models import (
    ReconciliationDecision,
    Relationship,
    TemporalFact,
    utc_now,
)


class RelationshipClassifier(Protocol):
    """Interface for pairwise runtime supersession classification."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify how a newly observed fact affects an existing fact."""
        ...


class _ClassificationOutput(BaseModel):
    """Structured semantic output before ledger identifiers are attached."""

    relationship: Relationship
    transition_time: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0)
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
"""

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """Configure LM Studio, optionally reusing an injected compatible client."""
        load_dotenv()
        if client is None:
            client = self._create_lmstudio_client()
        self._client = client
        self._model = model or os.environ["LMSTUDIO_MODEL"]

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
        """Request a structured pairwise decision from the configured model."""
        payload = {
            "old_fact": old_fact.model_dump(mode="json"),
            "new_fact": new_fact.model_dump(mode="json"),
        }
        response = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": self._INSTRUCTIONS},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            response_format=_ClassificationOutput,
            temperature=0.0,
        )
        output = response.choices[0].message.parsed
        if output is None:
            raise RuntimeError("relationship classifier returned no structured output")
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=output.relationship,
            transition_time=output.transition_time,
            confidence=output.confidence,
            rationale=output.rationale,
        )


class TemporalLedger:
    """Retain assertions and decisions while deriving their effective lifecycle."""

    def __init__(
        self,
        classifier: RelationshipClassifier,
        *,
        max_candidates: int | None = None,
    ) -> None:
        """Create an empty ledger with an injected semantic classifier."""
        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self._classifier = classifier
        self._max_candidates = max_candidates
        self._facts: dict[str, TemporalFact] = {}
        self._decisions: list[ReconciliationDecision] = []

    @property
    def facts(self) -> tuple[TemporalFact, ...]:
        """Return raw assertions in ingestion order."""
        return tuple(self._facts.values())

    @property
    def decisions(self) -> tuple[ReconciliationDecision, ...]:
        """Return reconciliation decisions in creation order."""
        return tuple(self._decisions)

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
            and (
                candidate.status == "active" or candidate.triple == new_fact.triple
            )
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

        decisions: list[ReconciliationDecision] = []
        knowledge_time = max(utc_now(), new_fact.observed_at)
        for old_fact in self.retrieve_related_facts(
            new_fact, known_at=knowledge_time
        ):
            decision = self._classifier.classify(old_fact, new_fact)
            self._validate_decision(decision, old_fact, new_fact)
            decisions.append(decision)

        self._facts[new_fact.fact_id] = new_fact
        self._decisions.extend(decisions)
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
        return tuple(resolved[fact_id] for fact_id in self._facts if fact_id in resolved)

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
        """Mark unresolved contradictory assertions as disputed."""
        for decision in decisions:
            if decision.relationship != "contradiction":
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
            if decision.relationship
            in {"state_transition", "correction", "duplicate"}
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
        boundary = max(matching, key=lambda decision: decision.decided_at).transition_time
        assert boundary is not None
        return boundary
