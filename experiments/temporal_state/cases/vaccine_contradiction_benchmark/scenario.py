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


# NEW OBSERVATION STRESS CASES: extend the original progression below this marker.
# Keep subjects and objects local to each pair so every gold decision tests one
# intended epistemic relationship rather than accidental cross-case retrieval.
_STRESS_CASES: tuple[_CaseSpec, ...] = (
    _CaseSpec(
        "duplicate-registry-report",
        _observation(
            "vax-stress-duplicate-registry-old",
            "Synthetic patient I immunization record",
            "records_administration_of",
            "Influenza vaccine on 2026-08-28",
            "The clinic registry records an influenza vaccination on August 28.",
            datetime(2026, 8, 28, tzinfo=UTC),
            20,
            EHR,
            # EXPIRATION STRESS EVENT: this evidence leaves the current view at minute 28.
            valid_to=KNOWLEDGE_START + timedelta(minutes=28),
        ),
        _observation(
            "vax-stress-duplicate-registry-new",
            "Synthetic patient I immunization record",
            "records_administration_of",
            "Influenza vaccine on 2026-08-28",
            "The public-health registry repeats the same dated administration event.",
            datetime(2026, 8, 28, tzinfo=UTC),
            21,
            CDC_TIMING_SOURCE,
        ),
        "duplicate",
        "duplicate",
        ("source_provenance",),
        "retain_both",
        "The sources independently encode the same administration event; retain both provenance records without duplicating the operational claim.",
    ),
    _CaseSpec(
        "correction-product-code",
        _observation(
            "vax-stress-correction-code-old",
            "Synthetic shipment J product identity",
            "has_product_code",
            "VX-1008",
            "An automated extraction transposes the final two digits of the product code.",
            datetime(2026, 8, 29, tzinfo=UTC),
            22,
            EXTRACTION,
        ),
        _observation(
            "vax-stress-correction-code-new",
            "Synthetic shipment J product identity",
            "has_product_code",
            "VX-1080",
            "The verified regulatory record corrects the extraction error to VX-1080.",
            datetime(2026, 8, 29, tzinfo=UTC),
            23,
            FDA,
        ),
        "correction",
        "correction",
        ("source_provenance", "domain_policy"),
        "accept_new",
        "The newer record explicitly corrects a documented extraction error rather than reporting a change in product identity.",
    ),
    _CaseSpec(
        "logical-storage-conflict",
        _observation(
            "vax-stress-storage-old",
            "Synthetic vaccine lot K storage disposition",
            "has_status",
            "Eligible for administration",
            "An unverified secondary summary says the temperature-excursion lot remains usable.",
            KNOWLEDGE_START + timedelta(minutes=24),
            24,
            NEWS,
        ),
        _observation(
            "vax-stress-storage-new",
            "Synthetic vaccine lot K storage disposition",
            "has_status",
            "Not eligible for administration",
            "The manufacturer-confirmed disposition rejects the same lot after the excursion.",
            KNOWLEDGE_START + timedelta(minutes=24),
            25,
            FDA,
            # EXPIRATION STRESS EVENT: the controlling disposition is time-bounded.
            valid_to=KNOWLEDGE_START + timedelta(minutes=30),
        ),
        "contradiction",
        "logical_contradiction",
        ("source_reliability", "source_provenance"),
        "accept_new",
        "The claims are mutually exclusive for the same lot and interval; the verified source-of-record disposition controls.",
    ),
    _CaseSpec(
        "temporal-inventory-change",
        _observation(
            "vax-stress-inventory-old",
            "Synthetic clinic L vaccine inventory",
            "has_status",
            "Out of stock",
            "The inventory system reports no available doses at opening time.",
            KNOWLEDGE_START + timedelta(minutes=26),
            26,
            EHR,
        ),
        _observation(
            "vax-stress-inventory-new",
            "Synthetic clinic L vaccine inventory",
            "has_status",
            "In stock",
            "A received shipment changes inventory availability later that morning.",
            KNOWLEDGE_START + timedelta(minutes=27),
            27,
            EHR,
        ),
        "state_transition",
        "temporal_change",
        ("source_recency", "temporal_validity"),
        "accept_new",
        "The inventory values describe successive valid states, so the later observation closes the earlier state.",
    ),
    _CaseSpec(
        "granularity-age-band",
        _observation(
            "vax-stress-age-band-old",
            "Synthetic vaccine M response estimate",
            "has_value",
            "82 percent in adults age 18 to 64",
            "A study reports the response estimate for adults aged 18 through 64.",
            datetime(2026, 7, 1, tzinfo=UTC),
            28,
            TRIAL,
            # EXPIRATION STRESS EVENT: the scoped estimate ages out of the current view.
            valid_to=KNOWLEDGE_START + timedelta(minutes=34),
        ),
        _observation(
            "vax-stress-age-band-new",
            "Synthetic vaccine M response estimate",
            "has_value",
            "61 percent in adults age 65 and older",
            "A companion analysis reports the response estimate for adults aged 65 and older.",
            datetime(2026, 7, 1, tzinfo=UTC),
            29,
            OBSERVATIONAL,
        ),
        "coexists",
        "granularity_mismatch",
        ("numerical_value", "temporal_validity"),
        "qualify",
        "The estimates use non-overlapping age populations and must remain qualified by age band.",
    ),
    _CaseSpec(
        "uncertain-cold-chain-gap",
        _observation(
            "vax-stress-cold-chain-old",
            "Synthetic lot N cold-chain assessment",
            "has_conclusion",
            "Temperature remained in range",
            "A partial logger export contains no recorded excursion but omits six hours.",
            KNOWLEDGE_START + timedelta(minutes=30),
            30,
            EXTRACTION,
        ),
        _observation(
            "vax-stress-cold-chain-new",
            "Synthetic lot N cold-chain assessment",
            "has_conclusion",
            "Temperature excursion possible",
            "A receiving note reports condensation but provides no calibrated temperature reading.",
            KNOWLEDGE_START + timedelta(minutes=30),
            31,
            NEWS,
        ),
        "uncertain",
        "uncertainty",
        ("source_reliability", "domain_policy"),
        "defer",
        "Neither incomplete record establishes the lot's condition; quarantine the conclusion for review while retaining both observations.",
    ),
    _CaseSpec(
        "coexist-coscheduled-products",
        _observation(
            "vax-stress-coschedule-old",
            "Synthetic encounter O planned products",
            "includes_product",
            "Influenza vaccine planned for encounter O",
            "The encounter plan includes influenza vaccination.",
            KNOWLEDGE_START + timedelta(minutes=32),
            32,
            EHR,
            # EXPIRATION STRESS EVENT: one planned product expires independently.
            valid_to=KNOWLEDGE_START + timedelta(minutes=38),
        ),
        _observation(
            "vax-stress-coschedule-new",
            "Synthetic encounter O planned products",
            "includes_product",
            "COVID-19 vaccine planned for encounter O",
            "The same encounter plan also includes COVID-19 vaccination.",
            KNOWLEDGE_START + timedelta(minutes=32),
            33,
            CDC_TIMING_SOURCE,
        ),
        "coexists",
        "compatible",
        ("relation_cardinality",),
        "retain_both",
        "The planning relation is multi-valued, so both products can remain visible for the encounter.",
    ),
    _CaseSpec(
        "logical-precaution-negation",
        _observation(
            "vax-stress-precaution-old",
            "Synthetic patient P same-day eligibility",
            "is_eligible_for",
            "Vaccine Y",
            "An intake form marks same-day eligibility before allergy verification.",
            KNOWLEDGE_START + timedelta(minutes=34),
            34,
            EHR,
        ),
        _observation(
            "vax-stress-precaution-new",
            "Synthetic patient P same-day eligibility",
            "is_not_eligible_for",
            "Vaccine Y",
            "The verified screening note explicitly blocks same-day administration pending review.",
            KNOWLEDGE_START + timedelta(minutes=34),
            35,
            EHR,
        ),
        "contradiction",
        "logical_contradiction",
        ("negation", "domain_policy"),
        "accept_new",
        "The positive and explicitly negated eligibility claims overlap in scope and time; the verified safety screen controls.",
    ),
    _CaseSpec(
        "temporal-deferral-cleared",
        _observation(
            "vax-stress-deferral-old",
            "Synthetic patient Q vaccination plan",
            "has_status",
            "Deferred pending symptom review",
            "The clinic records a temporary symptom-based deferral.",
            KNOWLEDGE_START + timedelta(minutes=36),
            36,
            EHR,
        ),
        _observation(
            "vax-stress-deferral-new",
            "Synthetic patient Q vaccination plan",
            "has_status",
            "Cleared for scheduling",
            "A later clinical review clears the temporary deferral and permits scheduling.",
            KNOWLEDGE_START + timedelta(minutes=37),
            37,
            EHR,
            # EXPIRATION STRESS EVENT: the cleared scheduling state is temporary.
            valid_to=KNOWLEDGE_START + timedelta(minutes=42),
        ),
        "state_transition",
        "temporal_change",
        ("temporal_validity",),
        "accept_new",
        "Deferral and clearance are sequential states; the clearance supersedes the earlier operational state.",
    ),
    _CaseSpec(
        "correction-dose-unit",
        _observation(
            "vax-stress-dose-unit-old",
            "Synthetic order R documented dose",
            "has_dose",
            "5 mL",
            "A machine extraction drops a decimal point and records 5 mL.",
            KNOWLEDGE_START + timedelta(minutes=38),
            38,
            EXTRACTION,
        ),
        _observation(
            "vax-stress-dose-unit-new",
            "Synthetic order R documented dose",
            "has_dose",
            "0.5 mL",
            "The source document is re-read and explicitly corrects the extracted value to 0.5 mL.",
            KNOWLEDGE_START + timedelta(minutes=38),
            39,
            FDA,
        ),
        "correction",
        "correction",
        ("numerical_value", "source_provenance"),
        "accept_new",
        "The change repairs a documented decimal extraction error; it is a correction, not a new dose interval.",
    ),
    _CaseSpec(
        "duplicate-synonymous-route",
        _observation(
            "vax-stress-route-old",
            "Synthetic administration S route",
            "has_route",
            "Intramuscular",
            "The administration record spells out the intramuscular route.",
            KNOWLEDGE_START + timedelta(minutes=40),
            40,
            EHR,
            # EXPIRATION STRESS EVENT: synonymous evidence is retired on schedule.
            valid_to=KNOWLEDGE_START + timedelta(minutes=46),
        ),
        _observation(
            "vax-stress-route-new",
            "Synthetic administration S route",
            "has_route",
            "Intramuscular",
            "A second export records the same route using a normalized vocabulary value.",
            KNOWLEDGE_START + timedelta(minutes=40),
            41,
            EXTRACTION,
        ),
        "duplicate",
        "duplicate",
        ("source_provenance",),
        "retain_both",
        "The records normalize to the same route claim and differ only in provenance and representation.",
    ),
    _CaseSpec(
        "granularity-geography",
        _observation(
            "vax-stress-geography-old",
            "Synthetic uptake estimate T",
            "has_value",
            "72 percent statewide",
            "A statewide dashboard reports aggregate vaccine uptake.",
            datetime(2026, 8, 31, tzinfo=UTC),
            42,
            CDC_TIMING_SOURCE,
        ),
        _observation(
            "vax-stress-geography-new",
            "Synthetic uptake estimate T",
            "has_value",
            "48 percent in District 7",
            "A district report gives uptake for one local jurisdiction during the same period.",
            datetime(2026, 8, 31, tzinfo=UTC),
            43,
            OBSERVATIONAL,
        ),
        "coexists",
        "granularity_mismatch",
        ("temporal_validity", "domain_policy"),
        "qualify",
        "Statewide and district estimates have different geographic denominators and should be retained with explicit scope.",
    ),
    _CaseSpec(
        "uncertain-adverse-event-causality",
        _observation(
            "vax-stress-causality-old",
            "Synthetic adverse event U assessment",
            "has_conclusion",
            "Unrelated to vaccination",
            "A spontaneous report labels the event unrelated without documenting an adjudication method.",
            KNOWLEDGE_START + timedelta(minutes=44),
            44,
            SOCIAL,
        ),
        _observation(
            "vax-stress-causality-new",
            "Synthetic adverse event U assessment",
            "has_conclusion",
            "Potentially related to vaccination",
            "A preliminary review calls the association possible but lacks complete clinical records.",
            KNOWLEDGE_START + timedelta(minutes=44),
            45,
            OBSERVATIONAL,
        ),
        "uncertain",
        "uncertainty",
        ("source_reliability", "temporal_validity"),
        "defer",
        "The evidence is incomplete on both sides and cannot support a causal winner without adjudication.",
    ),
    _CaseSpec(
        "logical-same-scope-dose",
        _observation(
            "vax-stress-dose-conflict-old",
            "Synthetic product V labeled presentation dose",
            "has_volume",
            "0.3 mL for presentation V",
            "An outdated secondary table lists 0.3 mL for the specified presentation and age group.",
            KNOWLEDGE_START + timedelta(minutes=46),
            46,
            NEWS,
        ),
        _observation(
            "vax-stress-dose-conflict-new",
            "Synthetic product V labeled presentation dose",
            "has_volume",
            "0.2 mL for presentation V",
            "The current regulator-approved label lists 0.2 mL for the same presentation and age group.",
            KNOWLEDGE_START + timedelta(minutes=46),
            47,
            FDA,
            # EXPIRATION STRESS EVENT: the selected label assertion later expires.
            valid_to=KNOWLEDGE_START + timedelta(minutes=50),
        ),
        "contradiction",
        "logical_contradiction",
        ("numerical_value", "domain_policy"),
        "accept_new",
        "The numerical claims share presentation, age, and interval; the regulator-approved label controls.",
    ),
    _CaseSpec(
        "coexist-distinct-endpoints",
        _observation(
            "vax-stress-endpoint-old",
            "Synthetic vaccine W study findings",
            "reports_outcome",
            "Reduced symptomatic disease",
            "The study reports a reduction in its symptomatic-disease endpoint.",
            KNOWLEDGE_START + timedelta(minutes=48),
            48,
            TRIAL,
        ),
        _observation(
            "vax-stress-endpoint-new",
            "Synthetic vaccine W study findings",
            "reports_outcome",
            "No measured reduction in asymptomatic infection",
            "The same study reports a separate result for asymptomatic infection.",
            KNOWLEDGE_START + timedelta(minutes=48),
            49,
            TRIAL,
        ),
        "coexists",
        "compatible",
        ("relation_cardinality", "domain_policy"),
        "retain_both",
        "The outcomes occupy distinct endpoints and do not negate one another.",
    ),
    _CaseSpec(
        "temporal-guidance-update",
        _observation(
            "vax-stress-guidance-old",
            "Synthetic campaign X recommended interval",
            "has_status",
            "Prior interval recommendation",
            "Archived campaign guidance records the interval used before the update.",
            KNOWLEDGE_START + timedelta(minutes=50),
            50,
            ARCHIVE,
        ),
        _observation(
            "vax-stress-guidance-new",
            "Synthetic campaign X recommended interval",
            "has_status",
            "Updated interval recommendation",
            "Current public-health guidance prospectively replaces the prior interval.",
            KNOWLEDGE_START + timedelta(minutes=51),
            51,
            CDC_TIMING_SOURCE,
        ),
        "state_transition",
        "temporal_change",
        ("source_recency", "temporal_validity"),
        "accept_new",
        "The guidance applies in successive validity intervals; preserve the archived recommendation and activate the update.",
    ),
)


# FIXED-POLICY GENERALIZATION CASES: add new observations below this marker
# without extending policy.json. These cases test whether the existing policy
# transfers to later, previously unseen evidence patterns.
_GENERALIZATION_CASES: tuple[_CaseSpec, ...] = (
    _CaseSpec(
        "generalization-logical-authority",
        _observation(
            "vax-general-logical-authority-old",
            "Synthetic lot Y release decision",
            "has_status",
            "Released for clinical use by local summary",
            "A local summary states that lot Y was released for clinical use.",
            KNOWLEDGE_START + timedelta(minutes=52),
            52,
            NEWS,
        ),
        _observation(
            "vax-general-logical-authority-new",
            "Synthetic lot Y release decision",
            "has_status",
            "Held from clinical use by regulator",
            "A regulator record places the same lot on hold for the same interval.",
            KNOWLEDGE_START + timedelta(minutes=52),
            53,
            FDA,
            valid_to=KNOWLEDGE_START + timedelta(minutes=76),
        ),
        "contradiction",
        "logical_contradiction",
        ("source_reliability", "source_provenance", "domain_policy"),
        "accept_new",
        "The release states are mutually exclusive in the same scope; the regulator record controls while preserving both sources.",
    ),
    _CaseSpec(
        "generalization-temporal-appointment",
        _observation(
            "vax-general-appointment-old",
            "Synthetic patient Z appointment state",
            "has_status",
            "Vaccination appointment scheduled",
            "The scheduling system records an appointment at the start of the interval.",
            KNOWLEDGE_START + timedelta(minutes=54),
            54,
            EHR,
        ),
        _observation(
            "vax-general-appointment-new",
            "Synthetic patient Z appointment state",
            "has_status",
            "Vaccination appointment completed",
            "The encounter record later marks that appointment completed.",
            KNOWLEDGE_START + timedelta(minutes=55),
            55,
            EHR,
        ),
        "state_transition",
        "temporal_change",
        ("source_recency", "temporal_validity"),
        "accept_new",
        "Scheduled and completed are successive appointment states rather than simultaneous claims.",
    ),
    _CaseSpec(
        "generalization-granularity-dose-number",
        _observation(
            "vax-general-dose-number-old",
            "Synthetic campaign AA dose-count estimate",
            "has_value",
            "Two-dose completion among enrolled adults",
            "A trial reports completion of a two-dose protocol among enrolled adults.",
            KNOWLEDGE_START + timedelta(minutes=56),
            56,
            TRIAL,
        ),
        _observation(
            "vax-general-dose-number-new",
            "Synthetic campaign AA dose-count estimate",
            "has_value",
            "One-dose uptake among all eligible residents",
            "A surveillance report measures first-dose uptake across all eligible residents.",
            KNOWLEDGE_START + timedelta(minutes=56),
            57,
            OBSERVATIONAL,
        ),
        "coexists",
        "granularity_mismatch",
        ("numerical_value", "domain_policy"),
        "qualify",
        "Dose number and population denominator differ, so both estimates require explicit scope qualifiers.",
    ),
    _CaseSpec(
        "generalization-duplicate-administration-time",
        _observation(
            "vax-general-admin-time-old",
            "Synthetic encounter AB administration timestamp",
            "occurred_at",
            "2026-09-01T09:15:00Z for encounter AB",
            "The clinic record timestamps administration at 09:15 UTC.",
            KNOWLEDGE_START + timedelta(minutes=58),
            58,
            EHR,
        ),
        _observation(
            "vax-general-admin-time-new",
            "Synthetic encounter AB administration timestamp",
            "occurred_at",
            "2026-09-01T09:15:00Z for encounter AB",
            "The registry export independently repeats the 09:15 UTC timestamp.",
            KNOWLEDGE_START + timedelta(minutes=58),
            59,
            CDC_TIMING_SOURCE,
        ),
        "duplicate",
        "duplicate",
        ("source_provenance",),
        "retain_both",
        "Both sources encode the same encounter timestamp; retain their provenance as one operational claim.",
    ),
    _CaseSpec(
        "generalization-correction-lot-suffix",
        _observation(
            "vax-general-lot-suffix-old",
            "Synthetic administration AC lot identifier",
            "has_lot_id",
            "LOT-AC-17B",
            "Optical character recognition records the lot suffix as 17B.",
            KNOWLEDGE_START + timedelta(minutes=60),
            60,
            EXTRACTION,
        ),
        _observation(
            "vax-general-lot-suffix-new",
            "Synthetic administration AC lot identifier",
            "has_lot_id",
            "LOT-AC-178",
            "A manual check of the source label corrects the final character from B to 8.",
            KNOWLEDGE_START + timedelta(minutes=60),
            61,
            EHR,
        ),
        "correction",
        "correction",
        ("source_provenance", "domain_policy"),
        "accept_new",
        "The second observation explicitly repairs an OCR error and does not describe a changing lot identity.",
    ),
    _CaseSpec(
        "generalization-uncertain-allergy-history",
        _observation(
            "vax-general-allergy-old",
            "Synthetic patient AD allergy assessment",
            "has_conclusion",
            "Prior vaccine allergy reported",
            "An intake form contains a patient-reported allergy without a named product or reaction.",
            KNOWLEDGE_START + timedelta(minutes=62),
            62,
            EHR,
        ),
        _observation(
            "vax-general-allergy-new",
            "Synthetic patient AD allergy assessment",
            "has_conclusion",
            "No vaccine allergy documented",
            "A record search finds no coded allergy but does not resolve the incomplete patient report.",
            KNOWLEDGE_START + timedelta(minutes=62),
            63,
            EXTRACTION,
        ),
        "uncertain",
        "uncertainty",
        ("source_reliability", "negation", "domain_policy"),
        "defer",
        "Missing coded documentation does not negate an underspecified report; retain both and require review.",
    ),
    _CaseSpec(
        "generalization-coexist-bilateral-sites",
        _observation(
            "vax-general-site-old",
            "Synthetic encounter AE administration sites",
            "includes_site",
            "Left deltoid for influenza dose AE",
            "The encounter records an influenza dose in the left deltoid.",
            KNOWLEDGE_START + timedelta(minutes=64),
            64,
            EHR,
        ),
        _observation(
            "vax-general-site-new",
            "Synthetic encounter AE administration sites",
            "includes_site",
            "Right deltoid for COVID dose AE",
            "The encounter also records a COVID dose in the right deltoid.",
            KNOWLEDGE_START + timedelta(minutes=64),
            65,
            EHR,
        ),
        "coexists",
        "compatible",
        ("relation_cardinality",),
        "retain_both",
        "The multi-valued administration-site relation can retain both product-specific sites.",
    ),
    _CaseSpec(
        "generalization-logical-explicit-negation",
        _observation(
            "vax-general-consent-old",
            "Synthetic encounter AF consent state",
            "has_status",
            "Consent granted for encounter AF",
            "An unsigned intake import marks consent as granted.",
            KNOWLEDGE_START + timedelta(minutes=66),
            66,
            EXTRACTION,
        ),
        _observation(
            "vax-general-consent-new",
            "Synthetic encounter AF consent state",
            "has_status",
            "Consent not granted for encounter AF",
            "The signed encounter record explicitly states that consent was not granted.",
            KNOWLEDGE_START + timedelta(minutes=66),
            67,
            EHR,
        ),
        "contradiction",
        "logical_contradiction",
        ("negation", "source_provenance", "domain_policy"),
        "accept_new",
        "The same-encounter consent states are mutually exclusive and the signed record controls.",
    ),
    _CaseSpec(
        "generalization-temporal-formulation-availability",
        _observation(
            "vax-general-formulation-old",
            "Synthetic clinic AG formulation availability",
            "has_status",
            "Legacy formulation available",
            "Inventory records show the legacy formulation available before replacement.",
            KNOWLEDGE_START + timedelta(minutes=68),
            68,
            ARCHIVE,
        ),
        _observation(
            "vax-general-formulation-new",
            "Synthetic clinic AG formulation availability",
            "has_status",
            "Updated formulation available",
            "A later receipt makes the updated formulation the current available stock.",
            KNOWLEDGE_START + timedelta(minutes=69),
            69,
            EHR,
            valid_to=KNOWLEDGE_START + timedelta(minutes=76),
        ),
        "state_transition",
        "temporal_change",
        ("source_recency", "temporal_validity"),
        "accept_new",
        "The formulations describe successive inventory states and the later receipt closes the legacy state.",
    ),
    _CaseSpec(
        "generalization-granularity-followup-window",
        _observation(
            "vax-general-followup-old",
            "Synthetic study AH safety summary",
            "reports_outcome",
            "No serious events during seven-day follow-up",
            "A short-window analysis reports no serious events in the first seven days.",
            KNOWLEDGE_START + timedelta(minutes=70),
            70,
            TRIAL,
        ),
        _observation(
            "vax-general-followup-new",
            "Synthetic study AH safety summary",
            "reports_outcome",
            "Rare serious events during six-month follow-up",
            "An extended analysis reports rare events over six months.",
            KNOWLEDGE_START + timedelta(minutes=70),
            71,
            OBSERVATIONAL,
        ),
        "coexists",
        "granularity_mismatch",
        ("temporal_validity", "numerical_value", "domain_policy"),
        "qualify",
        "The follow-up windows and event estimands differ, so both findings require temporal qualification.",
    ),
    _CaseSpec(
        "generalization-correction-administration-date",
        _observation(
            "vax-general-admin-date-old",
            "Synthetic record AI administration date",
            "occurred_on",
            "2026-08-11 for record AI",
            "A transcription import records the administration date as August 11.",
            KNOWLEDGE_START + timedelta(minutes=72),
            72,
            EXTRACTION,
        ),
        _observation(
            "vax-general-admin-date-new",
            "Synthetic record AI administration date",
            "occurred_on",
            "2026-08-17 for record AI",
            "The signed source record corrects the transcribed date to August 17.",
            KNOWLEDGE_START + timedelta(minutes=72),
            73,
            EHR,
        ),
        "correction",
        "correction",
        ("temporal_validity", "source_provenance"),
        "accept_new",
        "The signed source explicitly repairs a transcription error rather than asserting a second administration.",
    ),
    _CaseSpec(
        "generalization-duplicate-manufacturer-name",
        _observation(
            "vax-general-manufacturer-old",
            "Synthetic product AJ manufacturer identity",
            "has_manufacturer",
            "Example Biologics Incorporated for product AJ",
            "The label spells out the manufacturer's incorporated name.",
            KNOWLEDGE_START + timedelta(minutes=74),
            74,
            FDA,
            valid_to=KNOWLEDGE_START + timedelta(minutes=76),
        ),
        _observation(
            "vax-general-manufacturer-new",
            "Synthetic product AJ manufacturer identity",
            "has_manufacturer",
            "Example Biologics Incorporated for product AJ",
            "A normalized registry record resolves the same legal manufacturer name.",
            KNOWLEDGE_START + timedelta(minutes=74),
            75,
            CDC_TIMING_SOURCE,
        ),
        "duplicate",
        "duplicate",
        ("source_provenance",),
        "retain_both",
        "The sources resolve to the same legal entity and should not create duplicate operational claims.",
    ),
)


# NEW EXPIRATION STRESS SEQUENCE: standalone lifecycle observations belong here.
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


# FIXED-POLICY GENERALIZATION OBSERVATION: this independent event advances the
# ledger clock and forces all minute-76 expirations to reconcile before ingestion.
_GENERALIZATION_OBSERVATIONS = (
    _observation(
        "vax-general-expiration-sweep",
        "Synthetic benchmark expiration sweep AK",
        "records_status",
        "Minute 76 lifecycle boundary reached",
        "A synthetic audit marker advances knowledge time to the shared expiration boundary.",
        KNOWLEDGE_START + timedelta(minutes=76),
        76,
        EHR,
    ),
)


_ALL_CASES = (*_CASES, *_STRESS_CASES, *_GENERALIZATION_CASES)


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
    for case in _ALL_CASES
)
_CASE_BY_PAIR = {
    (case.old.fact_id, case.new.fact_id): case
    for case in _ALL_CASES
}
GOLD_LIFECYCLE_TRANSITIONS: tuple[GoldLifecycleTransition, ...] = (
    GoldLifecycleTransition(
        fact_id="vax-expiration-temporary",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-duplicate-registry-old",
        action="expire",
        preserve_history=True,
        review_required=False,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-storage-new",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-age-band-old",
        action="expire",
        preserve_history=True,
        review_required=False,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-coschedule-old",
        action="expire",
        preserve_history=True,
        review_required=False,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-deferral-new",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-route-old",
        action="expire",
        preserve_history=True,
        review_required=False,
    ),
    GoldLifecycleTransition(
        fact_id="vax-stress-dose-conflict-new",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
    GoldLifecycleTransition(
        fact_id="vax-general-logical-authority-new",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
    GoldLifecycleTransition(
        fact_id="vax-general-formulation-new",
        action="expire",
        preserve_history=True,
        review_required=True,
    ),
    GoldLifecycleTransition(
        fact_id="vax-general-manufacturer-old",
        action="expire",
        preserve_history=True,
        review_required=False,
    ),
)


def vaccine_observations() -> tuple[Observation, ...]:
    """Return paired and lifecycle observations in knowledge-time order."""
    paired = tuple(
        observation
        for case in _ALL_CASES
        for observation in (case.old, case.new)
    )
    return tuple(
        sorted(
            (*paired, *_EXPIRATION_OBSERVATIONS, *_GENERALIZATION_OBSERVATIONS),
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
            policy_version="1.2.0",
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
        review_by_fact = {
            "vax-expiration-temporary": True,
            "vax-stress-duplicate-registry-old": False,
            "vax-stress-storage-new": True,
            "vax-stress-age-band-old": False,
            "vax-stress-coschedule-old": False,
            "vax-stress-deferral-new": True,
            "vax-stress-route-old": False,
            "vax-stress-dose-conflict-new": True,
            "vax-general-logical-authority-new": True,
            "vax-general-formulation-new": True,
            "vax-general-manufacturer-old": False,
        }
        if fact.fact_id not in review_by_fact:
            raise KeyError(f"unknown vaccine expiration fact: {fact.fact_id}")
        return LifecycleResolution(
            review_required=review_by_fact[fact.fact_id],
            resolution=(
                "The observation reached its validity boundary. Remove it from the "
                "current graph, preserve its historical evidence, and apply the "
                "domain review requirement for the remaining operational state."
            ),
        )


def evaluate_vaccine_classifier(classifier: RelationshipClassifier) -> EvaluationReport:
    """Run a classifier and score each reasoning layer against independent gold."""
    ledger = TemporalLedger(classifier)
    facts = TemporalAnnotator().annotate_many(vaccine_observations())
    ledger.ingest_many(facts)
    return evaluate_decisions(
        gold_decisions=GOLD_DECISIONS,
        predictions=ledger.decisions,
        retained_fact_ids=(fact.fact_id for fact in ledger.facts),
        gold_lifecycle_transitions=GOLD_LIFECYCLE_TRANSITIONS,
        lifecycle_transitions=ledger.lifecycle_transitions,
    )
