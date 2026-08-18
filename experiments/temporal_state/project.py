"""Project temporal ledger history into ordinary KGGen graph snapshots."""

from __future__ import annotations

from datetime import datetime

from kg_gen.models import Graph

from experiments.temporal_state.ledger import TemporalLedger
from experiments.temporal_state.models import StateProjection, utc_now


def project_state(
    ledger: TemporalLedger,
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> StateProjection:
    """Build the state valid at one time using knowledge available at another."""
    valid_at = valid_at or utc_now()
    known_at = known_at or utc_now()
    facts = tuple(
        fact.model_copy(update={"status": "active"})
        if fact.status == "expired"
        else fact
        for fact in ledger.effective_facts(known_at=known_at)
        if fact.status not in {"retracted", "disputed"}
        and fact.is_valid_at(valid_at)
    )
    return StateProjection(
        valid_at=valid_at,
        known_at=known_at,
        facts=facts,
        relations=frozenset(fact.triple for fact in facts),
    )


def project_graph(
    ledger: TemporalLedger,
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> Graph:
    """Convert a bitemporal state snapshot to KGGen's static Graph model."""
    projection = project_state(ledger, valid_at=valid_at, known_at=known_at)
    return Graph(
        entities={
            entity
            for subject, _, object_ in projection.relations
            for entity in (subject, object_)
        },
        edges={predicate for _, predicate, _ in projection.relations},
        relations=set(projection.relations),
    )
