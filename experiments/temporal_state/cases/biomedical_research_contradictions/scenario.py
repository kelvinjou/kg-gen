"""A sourced bitemporal evidence graph for anti-amyloid Alzheimer research.

The fixture is synthetic, but its scientific and regulatory spine is factual.  It
models how a living review could ingest pivotal-trial results, regulatory actions,
label changes, coverage rules, extraction errors, and competing interpretations.
Deliberately erroneous or overgeneralized reports are included to exercise
correction and contradiction handling; they are not medical advice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from experiments.temporal_state.annotate import TemporalAnnotator
from experiments.temporal_state.ledger import TemporalLedger
from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    Relationship,
    TemporalFact,
)

UTC = timezone.utc
SCENARIO = "Living Anti-Amyloid Alzheimer Evidence Review"
KNOWLEDGE_START = datetime(2026, 1, 12, 14, 0, tzinfo=UTC)

EXPECTED_RELATIONSHIP_COUNTS: dict[Relationship, int] = {
    "duplicate": 5,
    "coexists": 5,
    "state_transition": 5,
    "correction": 5,
    "contradiction": 10,
    "uncertain": 0,
}


@dataclass(frozen=True)
class _ObservationSpec:
    """One historical assertion awaiting ingestion into the living review."""

    fact_id: str
    subject: str
    relation: str
    object: str
    valid_from: datetime
    source_text: str


def _date(year: int, month: int, day: int) -> datetime:
    """Return a midnight UTC validity timestamp."""
    return datetime(year, month, day, tzinfo=UTC)


# The order deliberately separates each contradiction from the report it opposes.
# Objects from upper layers reappear as subjects in lower layers, yielding paths such
# as evidence program -> trial node -> regimen/population -> safety question without
# causing the ledger's same-subject candidate retriever to create spurious pairs.
_SPECS: tuple[_ObservationSpec, ...] = (
    _ObservationSpec(
        "contra-aducanumab-efficacy-emerge",
        "Aducanumab phase 3 efficacy conclusion",
        "supports_clinical_benefit",
        "Yes: EMERGE high dose slowed CDR-SB decline by 22 percent",
        _date(2019, 10, 22),
        "The final EMERGE analysis met its primary endpoint (difference -0.39; P=.012). Source: Haeberlein et al., JPAD 2022, doi:10.14283/jpad.2022.30.",
    ),
    _ObservationSpec(
        "dup-leqembi-traditional-fda",
        "Leqembi traditional-approval provenance",
        "has_regulatory_event",
        "FDA traditional approval on 2023-07-06",
        _date(2023, 7, 6),
        "FDA announced conversion of Leqembi to traditional approval on July 6, 2023. Source: FDA traditional-approval announcement.",
    ),
    _ObservationSpec(
        "coex-program-clarity",
        SCENARIO,
        "contains_evidence_node",
        "CLARITY AD evidence node",
        _date(2022, 11, 29),
        "The living review added the pivotal lecanemab CLARITY AD trial as an evidence node. Source: van Dyck et al., NEJM 2023.",
    ),
    _ObservationSpec(
        "state-leqembi-accelerated",
        "Leqembi FDA approval-status timeline",
        "has_status",
        "Accelerated approval based on amyloid-plaque reduction",
        _date(2023, 1, 6),
        "FDA granted accelerated approval to Leqembi on January 6, 2023, using amyloid reduction as a surrogate endpoint. Source: FDA BLA 761269 materials.",
    ),
    _ObservationSpec(
        "corr-clarity-enrollment-1796",
        "CLARITY AD enrollment extraction",
        "reports_randomized_count",
        "1,796 participants",
        _date(2022, 11, 29),
        "An automated extraction incorrectly recorded 1,796 randomized participants from the CLARITY AD report.",
    ),
    _ObservationSpec(
        "contra-lecanemab-longterm-sustained",
        "Lecanemab benefit beyond 18 months",
        "has_conclusion",
        "Extension evidence establishes sustained clinical benefit beyond 18 months",
        _date(2022, 11, 29),
        "An extension analysis interpreted continued separation in clinical trajectories as evidence that lecanemab benefit persists beyond the blinded 18-month period.",
    ),
    _ObservationSpec(
        "dup-leqembi-traditional-letter",
        "Leqembi traditional-approval provenance",
        "has_regulatory_event",
        "FDA traditional approval on 2023-07-06",
        _date(2023, 7, 6),
        "The FDA supplemental approval letter independently records traditional approval on July 6, 2023. Source: FDA BLA 761269/S-001 approval letter.",
    ),
    _ObservationSpec(
        "coex-program-trailblazer",
        SCENARIO,
        "contains_evidence_node",
        "TRAILBLAZER-ALZ 2 evidence node",
        _date(2023, 7, 17),
        "The living review also added the pivotal donanemab TRAILBLAZER-ALZ 2 trial. Source: Sims et al., JAMA 2023.",
    ),
    _ObservationSpec(
        "contra-aducanumab-efficacy-engage",
        "Aducanumab phase 3 efficacy conclusion",
        "supports_clinical_benefit",
        "No: ENGAGE did not meet its primary CDR-SB endpoint",
        _date(2019, 10, 22),
        "The identically designed ENGAGE trial did not meet its primary endpoint (difference 0.03; P=.833). Source: Haeberlein et al., JPAD 2022.",
    ),
    _ObservationSpec(
        "state-leqembi-traditional",
        "Leqembi FDA approval-status timeline",
        "has_status",
        "Traditional approval after CLARITY AD verified clinical benefit",
        _date(2023, 7, 6),
        "FDA converted Leqembi to traditional approval after the confirmatory trial verified clinical benefit. Source: FDA, July 6, 2023.",
    ),
    _ObservationSpec(
        "corr-clarity-enrollment-1795",
        "CLARITY AD enrollment extraction",
        "reports_randomized_count",
        "1,795 participants",
        _date(2022, 11, 29),
        "Extraction correction: Study 301 enrolled 1,795 patients, not 1,796. Source: FDA traditional-approval announcement, July 6, 2023.",
    ),
    _ObservationSpec(
        "contra-lecanemab-longterm-unproven",
        "Lecanemab benefit beyond 18 months",
        "has_conclusion",
        "Sustained clinical benefit beyond 18 months is not established without controlled follow-up",
        _date(2022, 11, 29),
        "A methodological review concluded that open-label extension trajectories cannot establish durability beyond 18 months because the randomized placebo comparison ended.",
    ),
    _ObservationSpec(
        "contra-aducanumab-evidence-fda",
        "Aducanumab clinical-evidence sufficiency",
        "has_conclusion",
        "Amyloid reduction is reasonably likely to predict clinical benefit",
        _date(2021, 6, 7),
        "FDA leadership used robust plaque reduction as a surrogate reasonably likely to predict benefit for accelerated approval. Source: FDA decisional memorandum, 2021.",
    ),
    _ObservationSpec(
        "dup-clarity-id-registry",
        "CLARITY AD registration provenance",
        "has_registry_identifier",
        "ClinicalTrials.gov NCT03887455",
        _date(2019, 3, 25),
        "ClinicalTrials.gov registered the confirmatory lecanemab study as NCT03887455.",
    ),
    _ObservationSpec(
        "coex-clarity-regimen",
        "CLARITY AD evidence node",
        "tested_regimen",
        "Lecanemab 10 mg/kg intravenously every two weeks",
        _date(2019, 3, 21),
        "CLARITY AD randomized participants to lecanemab 10 mg/kg every two weeks or placebo. Source: NCT03887455 and NEJM 2023.",
    ),
    _ObservationSpec(
        "state-aduhelm-approved",
        "Aduhelm FDA approval-status timeline",
        "has_status",
        "Accelerated approval",
        _date(2021, 6, 7),
        "FDA granted Aduhelm accelerated approval on June 7, 2021. Source: FDA accelerated-approval records.",
    ),
    _ObservationSpec(
        "corr-clarity-duration-12",
        "CLARITY AD primary-endpoint extraction",
        "has_followup_horizon",
        "12 months",
        _date(2022, 11, 29),
        "A screening spreadsheet incorrectly normalized the CLARITY AD primary endpoint to a 12-month horizon.",
    ),
    _ObservationSpec(
        "contra-donanemab-transportability-supported",
        "TRAILBLAZER-ALZ 2 population transportability",
        "has_conclusion",
        "The treatment effect generalizes to racially and ethnically diverse populations",
        _date(2023, 7, 17),
        "An evidence review interpreted the randomized treatment effect and lack of a demonstrated subgroup interaction as supporting broad population transportability.",
    ),
    _ObservationSpec(
        "dup-clarity-id-publication",
        "CLARITY AD registration provenance",
        "has_registry_identifier",
        "ClinicalTrials.gov NCT03887455",
        _date(2019, 3, 25),
        "The pivotal publication independently identifies CLARITY AD as NCT03887455. Source: van Dyck et al., NEJM 2023.",
    ),
    _ObservationSpec(
        "coex-clarity-population",
        "CLARITY AD evidence node",
        "enrolled_population",
        "Amyloid-positive mild cognitive impairment or mild Alzheimer dementia",
        _date(2019, 3, 21),
        "The trial enrolled amyloid-positive participants with early Alzheimer disease. Source: NCT03887455 and NEJM 2023.",
    ),
    _ObservationSpec(
        "contra-aducanumab-evidence-dissent",
        "Aducanumab clinical-evidence sufficiency",
        "has_conclusion",
        "Existing clinical data are insufficient to establish effectiveness",
        _date(2021, 6, 7),
        "FDA's Office of Biostatistics director dissented, concluding that evidence was insufficient for accelerated or traditional approval. Source: FDA decisional memorandum, 2021.",
    ),
    _ObservationSpec(
        "state-aduhelm-withdrawn",
        "Aduhelm FDA approval-status timeline",
        "has_status",
        "Approval withdrawn; no longer FDA-approved",
        _date(2024, 11, 1),
        "FDA lists Aduhelm's accelerated approval as withdrawn on November 1, 2024. Source: FDA withdrawn accelerated approvals table.",
    ),
    _ObservationSpec(
        "corr-clarity-duration-18",
        "CLARITY AD primary-endpoint extraction",
        "has_followup_horizon",
        "18 months",
        _date(2022, 11, 29),
        "Extraction correction: the primary CDR-SB comparison was at 18 months, not 12 months. Source: van Dyck et al., NEJM 2023.",
    ),
    _ObservationSpec(
        "contra-donanemab-transportability-unsupported",
        "TRAILBLAZER-ALZ 2 population transportability",
        "has_conclusion",
        "The treatment effect cannot be generalized to racially and ethnically diverse populations",
        _date(2023, 7, 17),
        "A competing evidence review treated the trial's 91.5% White enrollment as insufficient to support generalization to more diverse populations. Source: Sims et al., JAMA 2023.",
    ),
    _ObservationSpec(
        "contra-lecanemab-meaning-fda",
        "CLARITY AD clinical-meaningfulness conclusion",
        "has_conclusion",
        "The 18-month CDR-SB reduction was clinically meaningful",
        _date(2023, 7, 6),
        "FDA described the statistically significant CLARITY AD effect as clinically meaningful when granting traditional approval. Source: FDA, July 6, 2023.",
    ),
    _ObservationSpec(
        "dup-donanemab-approval-fda",
        "Kisunla approval-date provenance",
        "has_regulatory_event",
        "FDA approval dated 2024-07-02",
        _date(2024, 7, 2),
        "The FDA approval package records Kisunla's original approval date as July 2, 2024. Source: BLA 761248 approval package.",
    ),
    _ObservationSpec(
        "coex-donanemab-efficacy",
        "TRAILBLAZER-ALZ 2 evidence node",
        "reported_efficacy",
        "Combined-population CDR-SB difference -0.70 at 76 weeks",
        _date(2023, 7, 17),
        "The combined population had a CDR-SB treatment difference of -0.70 at week 76. Source: Sims et al., JAMA 2023.",
    ),
    _ObservationSpec(
        "state-leqembi-mri-2024",
        "Leqembi label MRI-monitoring timeline",
        "requires_schedule",
        "Baseline MRI and MRIs before infusions 5, 7, and 14",
        _date(2024, 1, 1),
        "The 2024 prescribing information required MRI monitoring before the 5th, 7th, and 14th infusions. Source: FDA Leqembi label.",
    ),
    _ObservationSpec(
        "corr-donanemab-sites-227",
        "TRAILBLAZER-ALZ 2 site-count extraction",
        "reports_site_count",
        "227 centers",
        _date(2023, 7, 17),
        "An OCR-derived evidence table incorrectly recorded 227 participating centers.",
    ),
    _ObservationSpec(
        "contra-headtohead-donanemab-superior",
        "Lecanemab versus donanemab comparative effectiveness",
        "has_conclusion",
        "Donanemab provides greater slowing of cognitive-functional decline than lecanemab",
        _date(2024, 7, 2),
        "A cross-trial analysis ranked donanemab higher after comparing reported effects from TRAILBLAZER-ALZ 2 and CLARITY AD.",
    ),
    _ObservationSpec(
        "dup-donanemab-approval-purplebook",
        "Kisunla approval-date provenance",
        "has_regulatory_event",
        "FDA approval dated 2024-07-02",
        _date(2024, 7, 2),
        "FDA's Purple Book independently records July 2, 2024 as Kisunla's original approval date.",
    ),
    _ObservationSpec(
        "coex-donanemab-aria",
        "TRAILBLAZER-ALZ 2 evidence node",
        "reported_safety_event",
        "ARIA-E in 24.0 percent of donanemab participants",
        _date(2023, 7, 17),
        "ARIA-E occurred in 205 of 860 donanemab participants (24.0%). Source: Sims et al., JAMA 2023.",
    ),
    _ObservationSpec(
        "contra-lecanemab-meaning-threshold",
        "CLARITY AD clinical-meaningfulness conclusion",
        "has_conclusion",
        "Clinical meaningfulness cannot be established from a validated CDR-SB threshold",
        _date(2023, 7, 6),
        "The pivotal publication stated that a definition of clinically meaningful CDR-SB effects had not been established. Source: van Dyck et al., NEJM 2023.",
    ),
    _ObservationSpec(
        "state-leqembi-mri-2025",
        "Leqembi label MRI-monitoring timeline",
        "requires_schedule",
        "Baseline MRI and MRIs before infusions 3, 5, 7, and 14",
        _date(2025, 8, 28),
        "FDA required an additional MRI before the 3rd infusion after postmarketing review. Source: FDA Drug Safety Communication, August 28, 2025.",
    ),
    _ObservationSpec(
        "corr-donanemab-sites-277",
        "TRAILBLAZER-ALZ 2 site-count extraction",
        "reports_site_count",
        "277 centers",
        _date(2023, 7, 17),
        "Extraction correction: the trial used 277 centers in eight countries, not 227. Source: Sims et al., JAMA 2023.",
    ),
    _ObservationSpec(
        "contra-headtohead-lecanemab-superior",
        "Lecanemab versus donanemab comparative effectiveness",
        "has_conclusion",
        "Lecanemab provides greater slowing of cognitive-functional decline than donanemab",
        _date(2024, 7, 2),
        "A separately adjusted cross-trial analysis ranked lecanemab higher after accounting for differences in populations, follow-up, and outcome scales.",
    ),
    _ObservationSpec(
        "contra-amyloid-surrogate-fda",
        "Amyloid-plaque reduction as clinical-benefit surrogate",
        "supports_class_level_conclusion",
        "Reduction is reasonably likely to predict clinical benefit",
        _date(2021, 6, 7),
        "FDA accepted amyloid-plaque reduction as reasonably likely to predict clinical benefit for accelerated approval. Source: FDA aducanumab decisional memorandum.",
    ),
    _ObservationSpec(
        "dup-cms-ncd-manual",
        "CMS anti-amyloid NCD provenance",
        "has_effective_date",
        "NCD 200.3 effective 2022-04-07",
        _date(2022, 4, 7),
        "The Medicare NCD manual gives April 7, 2022 as the effective date for NCD 200.3. Source: CMS Transmittal 11692.",
    ),
    _ObservationSpec(
        "coex-label-population",
        "Lecanemab regimen node",
        "has_initiation_population",
        "Mild cognitive impairment or mild dementia stage of Alzheimer disease",
        _date(2023, 7, 6),
        "Leqembi should be initiated in the early-disease population studied in trials. Source: FDA prescribing information.",
    ),
    _ObservationSpec(
        "state-donanemab-complete-response",
        "Donanemab FDA review-status timeline",
        "has_status",
        "Complete response: insufficient 12-month exposure data",
        _date(2023, 1, 18),
        "FDA issued a complete response for the accelerated-approval submission because too few participants had at least 12 months of exposure. Source: FDA review and Lilly notice, January 2023.",
    ),
    _ObservationSpec(
        "corr-donanemab-amyloid-30",
        "TRAILBLAZER-ALZ 2 amyloid-threshold extraction",
        "has_screening_threshold",
        "At least 30 Centiloids",
        _date(2023, 7, 17),
        "A harmonization script incorrectly substituted a 30-Centiloid convention for the trial's screening threshold.",
    ),
    _ObservationSpec(
        "contra-anticoagulant-causal-risk",
        "Leqembi anticoagulant-associated hemorrhage risk",
        "has_conclusion",
        "Concomitant anticoagulant use causally increases intracerebral hemorrhage risk",
        _date(2023, 7, 6),
        "A safety review interpreted the increased number of intracerebral hemorrhages among anticoagulant users receiving Leqembi as a causal treatment interaction. Source: FDA traditional-approval materials.",
    ),
    _ObservationSpec(
        "dup-cms-ncd-decision",
        "CMS anti-amyloid NCD provenance",
        "has_effective_date",
        "NCD 200.3 effective 2022-04-07",
        _date(2022, 4, 7),
        "The CMS national-coverage decision memorandum independently gives April 7, 2022 as the effective date.",
    ),
    _ObservationSpec(
        "coex-label-apoe",
        "Lecanemab regimen node",
        "requires_risk_assessment",
        "APOE-e4 testing before treatment to inform ARIA risk",
        _date(2023, 7, 6),
        "The label states that APOE-e4 testing should be performed before treatment because homozygotes have higher ARIA incidence. Source: FDA prescribing information.",
    ),
    _ObservationSpec(
        "contra-amyloid-surrogate-meta",
        "Amyloid-plaque reduction as clinical-benefit surrogate",
        "supports_class_level_conclusion",
        "Randomized-trial synthesis does not establish a reliably meaningful cognitive benefit",
        _date(2022, 9, 1),
        "An updated instrumental-variable meta-analysis questioned how strongly amyloid reduction translates into cognitive change. Source: Ackley et al., BMJ 2022.",
    ),
    _ObservationSpec(
        "state-donanemab-approved",
        "Donanemab FDA review-status timeline",
        "has_status",
        "Traditional approval for Kisunla",
        _date(2024, 7, 2),
        "FDA approved Kisunla after review of the phase 3 confirmatory evidence. Source: FDA BLA 761248 approval package, July 2, 2024.",
    ),
    _ObservationSpec(
        "corr-donanemab-amyloid-37",
        "TRAILBLAZER-ALZ 2 amyloid-threshold extraction",
        "has_screening_threshold",
        "At least 37 Centiloids",
        _date(2023, 7, 17),
        "Extraction correction: TRAILBLAZER-ALZ 2 required amyloid PET of at least 37 Centiloids. Source: Sims et al., JAMA 2023.",
    ),
    _ObservationSpec(
        "contra-anticoagulant-noncausal-risk",
        "Leqembi anticoagulant-associated hemorrhage risk",
        "has_conclusion",
        "Available trial data do not establish that anticoagulants causally increase intracerebral hemorrhage risk",
        _date(2023, 7, 6),
        "A competing safety review concluded that the small nonrandomized subgroup and sparse events cannot establish a causal anticoagulant interaction; the label advises caution. Source: FDA prescribing information.",
    ),
    _ObservationSpec(
        "contra-class-lecanemab",
        "Anti-amyloid antibody class efficacy",
        "supports_generalized_conclusion",
        "Amyloid-targeting treatment slows cognitive-functional decline",
        _date(2022, 11, 29),
        "CLARITY AD reported significantly less decline with lecanemab than placebo in early symptomatic disease. Source: van Dyck et al., NEJM 2023.",
    ),
    _ObservationSpec(
        "dup-leqembi-aria-box-label",
        "Leqembi ARIA-warning provenance",
        "has_warning",
        "Boxed warning for amyloid-related imaging abnormalities",
        _date(2023, 7, 6),
        "The Leqembi prescribing information includes a boxed warning for ARIA. Source: FDA label.",
    ),
    _ObservationSpec(
        "coex-cms-eligibility",
        "CMS NCD 200.3 evidence node",
        "covers_population_under_CED",
        "MCI due to Alzheimer disease or mild Alzheimer dementia with confirmed amyloid",
        _date(2022, 4, 7),
        "CMS NCD 200.3 defines this population for Coverage with Evidence Development. Source: CMS Transmittal 11692.",
    ),
    _ObservationSpec(
        "state-aduhelm-program-active",
        "Aduhelm development-program timeline",
        "has_status",
        "Postmarketing ENVISION confirmatory study planned",
        _date(2022, 3, 1),
        "Biogen submitted a phase 4 ENVISION protocol to meet the accelerated-approval requirement. Source: Biogen 2022 filing.",
    ),
    _ObservationSpec(
        "corr-aduhelm-withdrawal-jan",
        "Aduhelm withdrawal-date extraction",
        "has_withdrawal_date",
        "2024-01-31",
        _date(2024, 11, 1),
        "A curator incorrectly treated Biogen's January 31 commercialization announcement as the FDA withdrawal date.",
    ),
    _ObservationSpec(
        "contra-stage-extrapolation-supported",
        "Lecanemab initiation outside early symptomatic Alzheimer disease",
        "has_conclusion",
        "Mechanistic and biomarker evidence supports clinical benefit outside the studied disease stage",
        _date(2023, 7, 6),
        "A translational review concluded that target engagement and amyloid lowering justify extrapolating clinical benefit to earlier or later Alzheimer disease stages.",
    ),
    _ObservationSpec(
        "dup-leqembi-aria-box-fda",
        "Leqembi ARIA-warning provenance",
        "has_warning",
        "Boxed warning for amyloid-related imaging abnormalities",
        _date(2023, 7, 6),
        "FDA's traditional-approval announcement independently notes the boxed ARIA warning.",
    ),
    _ObservationSpec(
        "coex-cms-pathway",
        "CMS NCD 200.3 evidence node",
        "uses_evidence_pathway",
        "Trials for surrogate approvals; CMS-approved studies or registries for direct-benefit approvals",
        _date(2022, 4, 7),
        "NCD 200.3 distinguishes coverage pathways by whether approval rests on a surrogate or a direct measure of benefit. Source: CMS Transmittal 11692.",
    ),
    _ObservationSpec(
        "contra-class-solanezumab",
        "Anti-amyloid antibody class efficacy",
        "supports_generalized_conclusion",
        "Amyloid-targeting treatment does not necessarily slow cognitive decline",
        _date(2023, 7, 17),
        "In preclinical Alzheimer disease, solanezumab did not significantly change the PACC cognitive outcome versus placebo. Source: Sperling et al., NEJM 2023.",
    ),
    _ObservationSpec(
        "state-aduhelm-program-discontinued",
        "Aduhelm development-program timeline",
        "has_status",
        "Development, commercialization, and ENVISION discontinued",
        _date(2024, 1, 31),
        "Biogen discontinued Aduhelm development and commercialization and terminated ENVISION. Source: Biogen, January 31, 2024.",
    ),
    _ObservationSpec(
        "corr-aduhelm-withdrawal-nov",
        "Aduhelm withdrawal-date extraction",
        "has_withdrawal_date",
        "2024-11-01",
        _date(2024, 11, 1),
        "Extraction correction: FDA records November 1, 2024 as the withdrawal date; January 31 was Biogen's program announcement. Source: FDA withdrawn approvals table.",
    ),
    _ObservationSpec(
        "contra-stage-extrapolation-unsupported",
        "Lecanemab initiation outside early symptomatic Alzheimer disease",
        "has_conclusion",
        "Clinical benefit should not be extrapolated outside the studied disease stage",
        _date(2023, 7, 6),
        "A clinical evidence review concluded that target engagement is insufficient to infer clinical benefit where safety and effectiveness were not studied. Source: FDA prescribing information.",
    ),
)


PAIR_RELATIONSHIPS: dict[frozenset[str], Relationship] = {
    frozenset(
        {"dup-leqembi-traditional-fda", "dup-leqembi-traditional-letter"}
    ): "duplicate",
    frozenset({"dup-clarity-id-registry", "dup-clarity-id-publication"}): "duplicate",
    frozenset(
        {"dup-donanemab-approval-fda", "dup-donanemab-approval-purplebook"}
    ): "duplicate",
    frozenset({"dup-cms-ncd-manual", "dup-cms-ncd-decision"}): "duplicate",
    frozenset(
        {"dup-leqembi-aria-box-label", "dup-leqembi-aria-box-fda"}
    ): "duplicate",
    frozenset({"coex-program-clarity", "coex-program-trailblazer"}): "coexists",
    frozenset({"coex-clarity-regimen", "coex-clarity-population"}): "coexists",
    frozenset({"coex-donanemab-efficacy", "coex-donanemab-aria"}): "coexists",
    frozenset({"coex-label-population", "coex-label-apoe"}): "coexists",
    frozenset({"coex-cms-eligibility", "coex-cms-pathway"}): "coexists",
    frozenset({"state-leqembi-accelerated", "state-leqembi-traditional"}): "state_transition",
    frozenset({"state-aduhelm-approved", "state-aduhelm-withdrawn"}): "state_transition",
    frozenset({"state-leqembi-mri-2024", "state-leqembi-mri-2025"}): "state_transition",
    frozenset(
        {"state-donanemab-complete-response", "state-donanemab-approved"}
    ): "state_transition",
    frozenset(
        {"state-aduhelm-program-active", "state-aduhelm-program-discontinued"}
    ): "state_transition",
    frozenset({"corr-clarity-enrollment-1796", "corr-clarity-enrollment-1795"}): "correction",
    frozenset({"corr-clarity-duration-12", "corr-clarity-duration-18"}): "correction",
    frozenset({"corr-donanemab-sites-227", "corr-donanemab-sites-277"}): "correction",
    frozenset(
        {"corr-donanemab-amyloid-30", "corr-donanemab-amyloid-37"}
    ): "correction",
    frozenset({"corr-aduhelm-withdrawal-jan", "corr-aduhelm-withdrawal-nov"}): "correction",
    frozenset(
        {"contra-aducanumab-efficacy-emerge", "contra-aducanumab-efficacy-engage"}
    ): "contradiction",
    frozenset(
        {"contra-aducanumab-evidence-fda", "contra-aducanumab-evidence-dissent"}
    ): "contradiction",
    frozenset(
        {"contra-lecanemab-meaning-fda", "contra-lecanemab-meaning-threshold"}
    ): "contradiction",
    frozenset(
        {"contra-amyloid-surrogate-fda", "contra-amyloid-surrogate-meta"}
    ): "contradiction",
    frozenset({"contra-class-lecanemab", "contra-class-solanezumab"}): "contradiction",
    frozenset(
        {"contra-lecanemab-longterm-sustained", "contra-lecanemab-longterm-unproven"}
    ): "contradiction",
    frozenset(
        {
            "contra-donanemab-transportability-supported",
            "contra-donanemab-transportability-unsupported",
        }
    ): "contradiction",
    frozenset(
        {"contra-headtohead-donanemab-superior", "contra-headtohead-lecanemab-superior"}
    ): "contradiction",
    frozenset(
        {"contra-anticoagulant-causal-risk", "contra-anticoagulant-noncausal-risk"}
    ): "contradiction",
    frozenset(
        {"contra-stage-extrapolation-supported", "contra-stage-extrapolation-unsupported"}
    ): "contradiction",
}

_RATIONALES: dict[Relationship, str] = {
    "duplicate": "Independent sources encode the same logical assertion.",
    "coexists": "The findings add compatible dimensions to the same evidence node.",
    "state_transition": "The later valid-time assertion supersedes an earlier historical state.",
    "correction": "The later provenance note explicitly repairs an extraction error.",
    "contradiction": "Credible records support incompatible conclusions for one scientific claim.",
    "uncertain": "Available evidence is insufficient to select a more decisive relationship.",
}


def biomedical_evidence_observations() -> tuple[Observation, ...]:
    """Return 60 validity-dated reports in deterministic knowledge-time order."""
    return tuple(
        Observation(
            fact_id=spec.fact_id,
            subject=spec.subject,
            relation=spec.relation,
            object=spec.object,
            source_text=spec.source_text,
            valid_from=spec.valid_from,
            observed_at=KNOWLEDGE_START + timedelta(minutes=10 * index),
        )
        for index, spec in enumerate(_SPECS)
    )


class BiomedicalEvidenceOracleClassifier:
    """Return the expected relationship for each curated evidence pair."""

    def classify(
        self, old_fact: TemporalFact, new_fact: TemporalFact
    ) -> ReconciliationDecision:
        """Classify a controlled pair without invoking a language model."""
        relationship = PAIR_RELATIONSHIPS.get(
            frozenset({old_fact.fact_id, new_fact.fact_id}), "coexists"
        )
        transition_time = (
            new_fact.valid_from if relationship == "state_transition" else None
        )
        return ReconciliationDecision(
            old_fact_id=old_fact.fact_id,
            new_fact_id=new_fact.fact_id,
            relationship=relationship,
            transition_time=transition_time,
            rationale=_RATIONALES[relationship],
            decided_at=new_fact.observed_at,
        )


def build_biomedical_evidence_ledger() -> TemporalLedger:
    """Ingest the complete scenario and return its reconciled temporal ledger."""
    ledger = TemporalLedger(BiomedicalEvidenceOracleClassifier())
    annotator = TemporalAnnotator()
    ledger.ingest_many(annotator.annotate_many(biomedical_evidence_observations()))
    return ledger


def relationship_counts(ledger: TemporalLedger) -> dict[Relationship, int]:
    """Count reconciliation decisions in the canonical relationship order."""
    counts = Counter(decision.relationship for decision in ledger.decisions)
    return {
        relationship: counts[relationship]
        for relationship in EXPECTED_RELATIONSHIP_COUNTS
    }
