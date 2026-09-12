"""Synthetic vaccine cases with independent semantic and action gold labels.

The cases test reasoning structure, not clinical advice.  Their source hierarchy
and terminology are grounded in the cited FDA and CDC source-of-record documents;
the patient and extraction records are fictional controlled perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.evaluation import (
    EvaluationReport,
    GoldDecision,
    GoldLifecycleTransition,
    evaluate_decisions,
)
from experiments.temporal_state.ledger import RelationshipClassifier, TemporalLedger
from experiments.temporal_state.models import (
    ContradictionDimension,
    EvidenceResolution,
    LifecycleResolution,
    IssueType,
    Observation,
    Provenance,
    ReconciliationDecision,
    Relationship,
    ResolutionAction,
    TemporalFact,
)

UTC = timezone.utc
KNOWLEDGE_START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
CDC_TIMING = "https://www.cdc.gov/vaccines/hcp/imz-best-practices/timing-spacing-immunobiologics.html"
CDC_CONTRAINDICATIONS = "https://www.cdc.gov/vaccines/hcp/imz-best-practices/contraindications-precautions.html"
FDA_COMIRNATY_LABEL = "https://www.fda.gov/media/151707/download"


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    old: Observation
    new: Observation
    relationship: Relationship
    issue_type: IssueType
    dimensions: tuple[ContradictionDimension, ...]
    action: ResolutionAction
    rationale: str


def _provenance(
    source_id: str,
    source_type: str,
    publisher: str,
    uri: str | None,
    reliability: float,
    quality: float,
) -> Provenance:
    return Provenance(
        source_id=source_id,
        source_type=source_type,
        publisher=publisher,
        uri=uri,
        reliability=reliability,
        methodological_quality=quality,
    )


def _observation(
    fact_id: str,
    subject: str,
    relation: str,
    object_: str,
    source_text: str,
    valid_from: datetime,
    observed_minute: int,
    provenance: Provenance,
    *,
    valid_to: datetime | None = None,
    confidence: float = 0.95,
) -> Observation:
    return Observation(
        fact_id=fact_id,
        subject=subject,
        relation=relation,
        object=object_,
        source_text=source_text,
        valid_from=valid_from,
        valid_to=valid_to,
        observed_at=KNOWLEDGE_START + timedelta(minutes=observed_minute),
        provenance=provenance,
        extraction_confidence=confidence,
    )


CDC = _provenance(
    "cdc-best-practices",
    "public-health-guideline",
    "CDC",
    CDC_CONTRAINDICATIONS,
    0.99,
    0.95,
)
CDC_TIMING_SOURCE = _provenance(
    "cdc-timing", "public-health-guideline", "CDC", CDC_TIMING, 0.99, 0.95
)
FDA = _provenance(
    "fda-comirnaty-label", "regulatory-label", "FDA", FDA_COMIRNATY_LABEL, 1.0, 0.98
)
ARCHIVE = _provenance(
    "archived-guidance", "archived-guideline", "CDC archive", None, 0.95, 0.9
)
TRIAL = _provenance(
    "pivotal-trial", "randomized-trial", "Trial investigators", None, 0.92, 0.95
)
OBSERVATIONAL = _provenance(
    "variant-effectiveness-study",
    "observational-study",
    "Public health network",
    None,
    0.9,
    0.82,
)
NEWS = _provenance("news-summary", "secondary-news", "Example News", None, 0.55, 0.4)
SOCIAL = _provenance("social-post", "social-media", "Unknown account", None, 0.15, 0.1)
EHR = _provenance(
    "synthetic-ehr", "clinical-record", "Synthetic clinic", None, 0.9, 0.85
)
EXTRACTION = _provenance(
    "automated-extraction", "machine-extraction", "Benchmark pipeline", None, 0.5, 0.4
)


_CASES: tuple[_CaseSpec, ...] = (
    _CaseSpec(
        "source-reliability",
        _observation(
            "vax-reliability-old",
            "Synthetic patient A influenza eligibility",
            "has_status",
            "Contraindicated because of mild illness",
            "An unattributed post says mild illness prohibits influenza vaccination.",
            datetime(2026, 8, 1, tzinfo=UTC),
            0,
            SOCIAL,
        ),
        _observation(
            "vax-reliability-new",
            "Synthetic patient A influenza eligibility",
            "has_status",
            "Not contraindicated by mild illness alone",
            "CDC distinguishes valid contraindications from conditions commonly misperceived as contraindications.",
            datetime(2026, 8, 1, tzinfo=UTC),
            1,
            CDC,
        ),
        "contradiction",
        "logical_contradiction",
        ("source_reliability", "source_provenance"),
        "accept_new",
        "The claims share scope and time; the source-of-record guidance controls over an unattributed post.",
    ),
    _CaseSpec(
        "source-recency",
        _observation(
            "vax-recency-old",
            "Seasonal vaccine formulation",
            "has_recommended_formulation",
            "Prior-season formulation",
            "An archived schedule describes the prior season's formulation.",
            datetime(2025, 8, 1, tzinfo=UTC),
            2,
            ARCHIVE,
            valid_to=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        _observation(
            "vax-recency-new",
            "Seasonal vaccine formulation",
            "has_recommended_formulation",
            "Current-season formulation",
            "The current schedule replaces the prior season's formulation prospectively.",
            datetime(2026, 8, 1, tzinfo=UTC),
            3,
            CDC_TIMING_SOURCE,
        ),
        "state_transition",
        "temporal_change",
        ("source_recency", "temporal_validity"),
        "accept_new",
        "Both claims can be historically true; only the newer one governs the current interval.",
    ),
    _CaseSpec(
        "temporal-validity",
        _observation(
            "vax-validity-old",
            "Synthetic patient B vaccination status",
            "has_status",
            "Dose due",
            "The dose was due before the vaccination encounter.",
            datetime(2026, 5, 1, tzinfo=UTC),
            4,
            EHR,
            valid_to=datetime(2026, 5, 10, tzinfo=UTC),
        ),
        _observation(
            "vax-validity-new",
            "Synthetic patient B vaccination status",
            "has_status",
            "Dose administered",
            "The record shows administration on May 10.",
            datetime(2026, 5, 10, tzinfo=UTC),
            5,
            EHR,
        ),
        "state_transition",
        "temporal_change",
        ("temporal_validity",),
        "accept_new",
        "Due and administered describe successive states, not a same-time contradiction.",
    ),
    _CaseSpec(
        "relation-cardinality",
        _observation(
            "vax-cardinality-old",
            "Synthetic encounter C",
            "administered_vaccine",
            "Influenza vaccine",
            "The encounter record lists influenza vaccine.",
            datetime(2026, 8, 20, tzinfo=UTC),
            6,
            EHR,
        ),
        _observation(
            "vax-cardinality-new",
            "Synthetic encounter C",
            "administered_vaccine",
            "COVID-19 vaccine",
            "The same encounter also lists COVID-19 vaccine; CDC best practices permit simultaneous administration of many vaccines.",
            datetime(2026, 8, 20, tzinfo=UTC),
            7,
            CDC_TIMING_SOURCE,
        ),
        "coexists",
        "compatible",
        ("relation_cardinality",),
        "retain_both",
        "administered_vaccine is multi-valued, so different objects can coexist.",
    ),
    _CaseSpec(
        "explicit-negation",
        _observation(
            "vax-negation-old",
            "Synthetic patient D vaccine eligibility",
            "is_eligible_for",
            "Vaccine X",
            "The clinic screening record marks the patient eligible.",
            datetime(2026, 8, 15, tzinfo=UTC),
            8,
            EHR,
        ),
        _observation(
            "vax-negation-new",
            "Synthetic patient D vaccine eligibility",
            "is_not_eligible_for",
            "Vaccine X",
            "A verified same-day allergy record explicitly negates eligibility pending specialist review.",
            datetime(2026, 8, 15, tzinfo=UTC),
            9,
            EHR,
        ),
        "contradiction",
        "logical_contradiction",
        ("negation", "domain_policy"),
        "accept_new",
        "Explicit negation in the same scope conflicts with the positive assertion; the safety policy favors the verified allergy record.",
    ),
    _CaseSpec(
        "numerical-granularity",
        _observation(
            "vax-number-old",
            "Vaccine effectiveness estimate",
            "has_value",
            "95 percent against symptomatic disease in trial population",
            "A pivotal trial reports efficacy for its protocol-defined population and endpoint.",
            datetime(2020, 11, 1, tzinfo=UTC),
            10,
            TRIAL,
        ),
        _observation(
            "vax-number-new",
            "Vaccine effectiveness estimate",
            "has_value",
            "50 percent against infection during a later variant period",
            "A later observational study estimates a different endpoint, population, and variant period.",
            datetime(2022, 1, 1, tzinfo=UTC),
            11,
            OBSERVATIONAL,
        ),
        "coexists",
        "granularity_mismatch",
        ("numerical_value", "temporal_validity"),
        "qualify",
        "The numbers are incomparable until endpoint, population, and variant period are attached.",
    ),
    _CaseSpec(
        "source-provenance",
        _observation(
            "vax-provenance-old",
            "COMIRNATY labeled dose",
            "has_volume",
            "0.2 mL",
            "A secondary story reports a 0.2 mL dose without identifying presentation or age group.",
            datetime(2026, 8, 27, tzinfo=UTC),
            12,
            NEWS,
        ),
        _observation(
            "vax-provenance-new",
            "COMIRNATY labeled dose",
            "has_volume",
            "Presentation-specific volume in current FDA label",
            "The current FDA prescribing information is the source of record; dose volume must be read with presentation and age.",
            datetime(2026, 8, 27, tzinfo=UTC),
            13,
            FDA,
        ),
        "contradiction",
        "logical_contradiction",
        ("source_provenance", "numerical_value", "domain_policy"),
        "accept_new",
        "The regulatory label controls dosing; the unscoped secondary number is unsafe to operationalize.",
    ),
    _CaseSpec(
        "uncertain-safety-signal",
        _observation(
            "vax-uncertainty-old",
            "Rare event causal association",
            "has_conclusion",
            "No causal association",
            "An underpowered analysis observes no statistically conclusive association.",
            datetime(2026, 1, 1, tzinfo=UTC),
            14,
            OBSERVATIONAL,
        ),
        _observation(
            "vax-uncertainty-new",
            "Rare event causal association",
            "has_conclusion",
            "Possible safety signal",
            "A passive surveillance analysis reports a signal but cannot establish causality.",
            datetime(2026, 1, 1, tzinfo=UTC),
            15,
            OBSERVATIONAL,
        ),
        "uncertain",
        "uncertainty",
        ("source_reliability", "domain_policy"),
        "defer",
        "Neither absence of significance nor a passive signal establishes the causal conclusion; preserve both for review.",
    ),
)


_EXPIRATION_OBSERVATIONS = (
    _observation(
        "vax-expiration-temporary",
        "Synthetic patient H temporary vaccination plan",
        "has_status",
        "Vaccination deferred during observation window",
        "A synthetic clinic places a time-bounded hold on vaccination while an observation window remains open.",
        KNOWLEDGE_START + timedelta(minutes=16),
        16,
        EHR,
        valid_to=KNOWLEDGE_START + timedelta(minutes=18),
    ),
    _observation(
        "vax-expiration-followup",
        "Synthetic patient H temporary vaccination plan",
        "has_status",
        "Routine eligibility reassessment required",
        "After the temporary hold expires, the synthetic clinic records that routine eligibility must be reassessed.",
        KNOWLEDGE_START + timedelta(minutes=19),
        19,
        EHR,
    ),
)


GOLD_DECISIONS: tuple[GoldDecision, ...] = tuple(
    GoldDecision(
        case_id=case.case_id,
        old_fact_id=case.old.fact_id or "",
        new_fact_id=case.new.fact_id or "",
        issue_type=case.issue_type,
        contradiction_dimensions=case.dimensions,
        expected_action=case.action,
        rationale=case.rationale,
    )
    for case in _CASES
)
_CASE_BY_PAIR = {(case.old.fact_id, case.new.fact_id): case for case in _CASES}
GOLD_LIFECYCLE_TRANSITIONS: tuple[GoldLifecycleTransition, ...] = (
    GoldLifecycleTransition(
        fact_id="vax-expiration-temporary",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
)


def vaccine_observations() -> tuple[Observation, ...]:
    """Return paired and lifecycle observations in knowledge-time order."""
    paired = tuple(
        observation for case in _CASES for observation in (case.old, case.new)
    )
    return tuple(
        sorted(
            (*paired, *_EXPIRATION_OBSERVATIONS),
            key=lambda item: item.observed_at,
        )
    )


class VaccineGoldClassifier:
    """Deterministic reference implementation, separate from evaluated systems."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        case = _CASE_BY_PAIR[(old_fact.fact_id, new_fact.fact_id)]
        selected = {
            "accept_old": "old",
            "accept_new": "new",
        }.get(case.action)
        qualified = (
            "Report both estimates with endpoint, population, and time qualifiers."
            if case.action == "qualify"
            else None
        )
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=case.relationship,
            issue_type=case.issue_type,
            contradiction_dimensions=case.dimensions,
            transition_time=(
                new_fact.valid_from if case.relationship == "state_transition" else None
            ),
            rationale=case.rationale,
            policy_name="vaccine_claim_resolution",
            policy_version="1.1.0",
            policy_applied_for=case.relationship,
            policy_resolution=EvidenceResolution(
                action=case.action,
                selected_evidence=selected,
                qualified_assertion=qualified,
                review_required=case.action in {"defer", "qualify"},
                resolution=case.rationale,
            ),
            decided_at=new_fact.observed_at,
        )

    def resolve_expiration(self, fact: TemporalFact) -> LifecycleResolution:
        """Return the controlled domain outcome for the expiration benchmark."""
        if fact.fact_id != "vax-expiration-temporary":
            raise KeyError(f"unknown vaccine expiration fact: {fact.fact_id}")
        return LifecycleResolution(
            review_required=True,
            resolution=(
                "The temporary hold expired. Preserve its historical evidence and "
                "require routine eligibility reassessment before operational use."
            ),
        )


def evaluate_vaccine_classifier(classifier: RelationshipClassifier) -> EvaluationReport:
    """Run a classifier and score each reasoning layer against independent gold."""
    ledger = TemporalLedger(classifier)
    facts = TemporalAnnotator().annotate_many(vaccine_observations())
    ledger.ingest_many(facts)
    return evaluate_decisions(
        GOLD_DECISIONS,
        ledger.decisions,
        retained_fact_ids=(fact.fact_id for fact in ledger.facts),
        gold_lifecycle_transitions=GOLD_LIFECYCLE_TRANSITIONS,
        lifecycle_transitions=ledger.lifecycle_transitions,
    )
