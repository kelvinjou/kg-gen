"""Twenty interleaved emergency observations with delayed contradictions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    TemporalFact,
)

UTC = timezone.utc
INCIDENT = "Regional Emergency 2026-10-12"
START = datetime(2026, 10, 12, 8, 0, tzinfo=UTC)


def _observation(
    index: int,
    fact_id: str,
    subject: str,
    relation: str,
    object_: str,
    source_text: str,
    *,
    valid_index: int | None = None,
) -> Observation:
    """Build one observation at a five-minute timeline interval."""
    effective_index = index if valid_index is None else valid_index
    return Observation(
        fact_id=fact_id,
        subject=subject,
        relation=relation,
        object=object_,
        source_text=source_text,
        valid_from=START + timedelta(minutes=5 * effective_index),
        observed_at=START + timedelta(minutes=5 * index),
    )


def nonconsecutive_contradiction_observations() -> tuple[Observation, ...]:
    """Return 20 reports containing eight non-adjacent contradictions."""
    return (
        _observation(
            0,
            "north-zone-clear",
            "North Evacuation Zone",
            "evacuation_status",
            "No evacuation ordered",
            "A municipal desk report said no evacuation was ordered for North Zone.",
        ),
        _observation(
            1,
            "weather-heavy-rain",
            "Weather Station Alpha",
            "reported_condition",
            "Heavy rain",
            "Weather Station Alpha measured heavy rain across the response area.",
        ),
        _observation(
            2,
            "route-6-open",
            "Route 6",
            "access_status",
            "Open",
            "A transportation dispatcher reported Route 6 open at 08:10.",
        ),
        _observation(
            3,
            "north-zone-evacuate",
            "North Evacuation Zone",
            "evacuation_status",
            "Mandatory evacuation ordered",
            "A field command post reported a mandatory North Zone evacuation at 08:00.",
            valid_index=0,
        ),
        _observation(
            4,
            "shelter-delta-open",
            "Shelter Delta",
            "operating_status",
            "Open",
            "The shelter coordination desk listed Shelter Delta as open.",
        ),
        _observation(
            5,
            "reservoir-capacity-high",
            "Emergency Reservoir",
            "available_capacity",
            "80 percent",
            "The utilities dashboard reported 80 percent emergency capacity.",
        ),
        _observation(
            6,
            "route-6-closed",
            "Route 6",
            "access_status",
            "Closed",
            "A road patrol report said Route 6 was closed at 08:10.",
            valid_index=2,
        ),
        _observation(
            7,
            "radio-network-operational",
            "Emergency Radio Network",
            "communications_status",
            "Operational",
            "A radio check confirmed the emergency communications network was operational.",
        ),
        _observation(
            8,
            "east-bridge-passable",
            "East Bridge",
            "safety_status",
            "Passable",
            "An infrastructure desk report listed East Bridge as passable.",
        ),
        _observation(
            9,
            "shelter-delta-closed",
            "Shelter Delta",
            "operating_status",
            "Closed",
            "A shelter field team reported Shelter Delta closed at 08:20.",
            valid_index=4,
        ),
        _observation(
            10,
            "district-water-safe",
            "District Water System",
            "potability_status",
            "Safe to drink",
            "The operations dashboard reported district water safe to drink.",
        ),
        _observation(
            11,
            "reservoir-capacity-low",
            "Emergency Reservoir",
            "available_capacity",
            "20 percent",
            "A reservoir technician reported only 20 percent capacity at 08:25.",
            valid_index=5,
        ),
        _observation(
            12,
            "ridge-wind-northeast",
            "Wind Sensor Ridge",
            "reported_wind",
            "35 mph northeast",
            "The ridge sensor measured northeast winds at 35 mph.",
        ),
        _observation(
            13,
            "east-bridge-unsafe",
            "East Bridge",
            "safety_status",
            "Unsafe",
            "A structural inspection team reported East Bridge unsafe at 08:40.",
            valid_index=8,
        ),
        _observation(
            14,
            "county-hospital-operational",
            "County Hospital",
            "operating_status",
            "Operational",
            "The regional dashboard listed County Hospital as operational.",
        ),
        _observation(
            15,
            "district-water-unsafe",
            "District Water System",
            "potability_status",
            "Unsafe to drink",
            "A sampling team reported district water unsafe at 08:50.",
            valid_index=10,
        ),
        _observation(
            16,
            "south-zone-clear",
            "South Evacuation Zone",
            "evacuation_status",
            "No evacuation ordered",
            "A public information report said South Zone was not under evacuation.",
        ),
        _observation(
            17,
            "county-hospital-offline",
            "County Hospital",
            "operating_status",
            "Offline",
            "A medical coordination call reported County Hospital offline at 09:10.",
            valid_index=14,
        ),
        _observation(
            18,
            "supply-depot-ready",
            "Supply Depot West",
            "staging_status",
            "Ready",
            "The logistics lead reported Supply Depot West ready for distribution.",
        ),
        _observation(
            19,
            "south-zone-evacuate",
            "South Evacuation Zone",
            "evacuation_status",
            "Mandatory evacuation ordered",
            "A law-enforcement report said South Zone had a mandatory evacuation at 09:20.",
            valid_index=16,
        ),
    )


class NonconsecutiveContradictionOracleClassifier:
    """Deterministically identify the eight contradiction pairs in this fixture."""

    CONTRADICTION_PAIRS = frozenset(
        {
            frozenset({"north-zone-clear", "north-zone-evacuate"}),
            frozenset({"route-6-open", "route-6-closed"}),
            frozenset({"shelter-delta-open", "shelter-delta-closed"}),
            frozenset({"reservoir-capacity-high", "reservoir-capacity-low"}),
            frozenset({"east-bridge-passable", "east-bridge-unsafe"}),
            frozenset({"district-water-safe", "district-water-unsafe"}),
            frozenset({"county-hospital-operational", "county-hospital-offline"}),
            frozenset({"south-zone-clear", "south-zone-evacuate"}),
        }
    )

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify known opposing reports as contradictions."""
        pair = frozenset({old_fact.fact_id, new_fact.fact_id})
        relationship = (
            "contradiction" if pair in self.CONTRADICTION_PAIRS else "coexists"
        )
        rationale = (
            "The reports make mutually exclusive claims for one effective interval."
            if relationship == "contradiction"
            else "The reports describe compatible dimensions of the emergency."
        )
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=relationship,
            rationale=rationale,
            decided_at=new_fact.observed_at,
        )
