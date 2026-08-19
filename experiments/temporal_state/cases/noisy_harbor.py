"""Conflicting, corrected, and ambiguous reports about a harbor spill."""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    TemporalFact,
)

UTC = timezone.utc
INCIDENT = "Bayview Harbor Spill 2026-09-04"


def noisy_harbor_observations() -> tuple[Observation, ...]:
    """Return reports designed to exercise the non-happy-path labels."""
    return (
        Observation(
            fact_id="location-pier-7",
            subject=INCIDENT,
            relation="reported_source_at",
            object="Pier 7",
            valid_from=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 4, 9, 3, tzinfo=UTC),
            source_text=(
                "The Coast Guard reported that the spill originated at Pier 7 at 09:00."
            ),
        ),
        Observation(
            fact_id="location-pier-9",
            subject=INCIDENT,
            relation="reported_source_at",
            object="Pier 9",
            valid_from=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
            observed_at=datetime(2026, 9, 4, 9, 8, tzinfo=UTC),
            source_text=(
                "The harbor operator disputed the Coast Guard report and said "
                "the spill originated at Pier 9, not Pier 7, at 09:00."
            ),
        ),
        Observation(
            fact_id="volume-5000",
            subject=INCIDENT,
            relation="estimated_release",
            object="5,000 gallons",
            valid_from=datetime(2026, 9, 4, 9, 15, tzinfo=UTC),
            observed_at=datetime(2026, 9, 4, 9, 18, tzinfo=UTC),
            source_text=(
                "An initial city bulletin estimated that 5,000 gallons had "
                "been released."
            ),
        ),
        Observation(
            fact_id="volume-500-correction",
            subject=INCIDENT,
            relation="estimated_release",
            object="500 gallons",
            valid_from=datetime(2026, 9, 4, 9, 15, tzinfo=UTC),
            observed_at=datetime(2026, 9, 4, 9, 31, tzinfo=UTC),
            source_text=(
                "Correction: the city said the earlier bulletin contained an "
                "extra zero; the estimate was 500 gallons, not 5,000."
            ),
        ),
        Observation(
            fact_id="boom-a-deployed",
            subject="Containment Boom A",
            relation="deployed_at",
            object="East Channel",
            valid_from=datetime(2026, 9, 4, 9, 40, tzinfo=UTC),
            observed_at=datetime(2026, 9, 4, 9, 43, tzinfo=UTC),
            source_text=(
                "The response inventory recorded Containment Boom A as deployed "
                "at East Channel."
            ),
        ),
        Observation(
            fact_id="boom-alpha-recovered",
            subject="Boom Alpha",
            relation="recovered_from",
            object="East Channel",
            valid_from=datetime(2026, 9, 4, 9, 45, tzinfo=UTC),
            observed_at=datetime(2026, 9, 4, 9, 47, tzinfo=UTC),
            source_text=(
                "A field note said Boom Alpha was recovered from East Channel, "
                "but gave no serial number. Crews sometimes call Containment "
                "Boom A 'Alpha', and sometimes use 'Alpha' for a second boom, "
                "so it is unknown whether this is the deployed boom or another one."
            ),
        ),
    )


class NoisyHarborOracleClassifier:
    """Encode expected pairwise outcomes for the deliberately noisy fixture."""

    _LOCATION_FACTS = frozenset({"location-pier-7", "location-pier-9"})
    _VOLUME_FACTS = frozenset({"volume-5000", "volume-500-correction"})
    _AMBIGUOUS_BOOM_FACTS = frozenset({"boom-a-deployed", "boom-alpha-recovered"})

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Distinguish a dispute, an explicit correction, and weak evidence."""
        pair = frozenset({old_fact.fact_id, new_fact.fact_id})
        if pair == self._LOCATION_FACTS:
            relationship = "contradiction"
            rationale = "Two sources give mutually exclusive origins for one time."
        elif pair == self._VOLUME_FACTS:
            relationship = "correction"
            rationale = "The later city bulletin explicitly corrects a typo."
        elif pair == self._AMBIGUOUS_BOOM_FACTS:
            relationship = "uncertain"
            rationale = (
                "The evidence does not establish whether A and Alpha are one boom."
            )
        else:
            relationship = "coexists"
            rationale = "The reports describe independent dimensions of the incident."

        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=relationship,
            rationale=rationale,
            decided_at=new_fact.observed_at,
        )
