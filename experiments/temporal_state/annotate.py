"""Annotate KGGen relations with temporal and provenance metadata."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from kg_gen.models import Graph

from experiments.temporal_state.models import Observation, Relation, TemporalFact


class TemporalAnnotator:
    """Convert observations into temporal facts without reconciling them."""

    def annotate(self, observation: Observation) -> TemporalFact:
        """Create one immutable fact, defaulting valid time to observation time."""
        values = {
            "subject": observation.subject,
            "relation": observation.relation,
            "object": observation.object,
            "valid_from": observation.valid_from or observation.observed_at,
            "valid_to": observation.valid_to,
            "observed_at": observation.observed_at,
            "source_text": observation.source_text,
        }
        if observation.fact_id is not None:
            values["fact_id"] = observation.fact_id
        return TemporalFact.model_validate(values)

    def annotate_relation(
        self,
        relation: Relation,
        *,
        source_text: str,
        observed_at: datetime,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        fact_id: str | None = None,
    ) -> TemporalFact:
        """Annotate a KGGen relation tuple with shared temporal metadata."""
        subject, predicate, object_ = relation
        return self.annotate(
            Observation(
                fact_id=fact_id,
                subject=subject,
                relation=predicate,
                object=object_,
                source_text=source_text,
                observed_at=observed_at,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )

    def annotate_many(
        self, observations: Iterable[Observation]
    ) -> tuple[TemporalFact, ...]:
        """Annotate observations in input order."""
        return tuple(self.annotate(observation) for observation in observations)

    def annotate_graph(
        self,
        graph: Graph,
        *,
        source_text: str,
        observed_at: datetime,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> tuple[TemporalFact, ...]:
        """Annotate every relation in a static KGGen graph deterministically."""
        return tuple(
            self.annotate_relation(
                relation,
                source_text=source_text,
                observed_at=observed_at,
                valid_from=valid_from,
                valid_to=valid_to,
            )
            for relation in sorted(graph.relations)
        )
