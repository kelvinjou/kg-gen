"""Regression checks for temporal graph visualization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from kg_gen.models import Graph

from experiments.temporal_state.models import (
    EvidenceResolution,
    Observation,
    ReconciliationDecision,
)
from experiments.temporal_state.visualizer.visualize_kg import (
    open_output_visualization,
    visualize_snapshots,
)


def test_open_output_visualization_opens_existing_html(tmp_path: Path) -> None:
    """Open and return a resolved HTML file from the selected output folder."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    graph_html = output_dir / "graph.html"
    graph_html.write_text("<html></html>", encoding="utf-8")

    with patch(
        "experiments.temporal_state.visualizer.visualize_kg.webbrowser.open"
    ) as open_browser:
        destination = open_output_visualization(output_dir=output_dir)

    assert destination == graph_html.resolve()
    open_browser.assert_called_once_with(graph_html.resolve().as_uri())


def test_snapshot_timeline_accepts_an_empty_graph(tmp_path: Path) -> None:
    """Preserve an empty state caused by disputed facts in the timeline."""
    active = Graph(
        entities={"Incident", "Pier 7"},
        edges={"reported_at"},
        relations={("Incident", "reported_at", "Pier 7")},
    )
    empty = Graph(entities=set(), edges=set(), relations=set())

    destination = visualize_snapshots(
        (("initial report", active), ("claims disputed", empty)),
        str(tmp_path / "timeline.html"),
    )

    html = destination.read_text(encoding="utf-8")
    assert '"label": "claims disputed"' in html
    assert "No active relations in this snapshot" in html


def test_snapshot_timeline_serializes_decisions_and_edge_changes(
    tmp_path: Path,
) -> None:
    """Expose decision details and flag the replacement edge as updated."""
    initial = Graph(
        entities={"Incident", "5,000 gallons"},
        edges={"estimated_release"},
        relations={("Incident", "estimated_release", "5,000 gallons")},
    )
    corrected = Graph(
        entities={"Incident", "500 gallons"},
        edges={"estimated_release"},
        relations={("Incident", "estimated_release", "500 gallons")},
    )
    decision = ReconciliationDecision(
        old_fact_id="volume-5000",
        new_fact_id="volume-500-correction",
        relationship="correction",
        rationale="The later bulletin explicitly corrects the estimate.",
    )

    destination = visualize_snapshots(
        (
            ("initial estimate", initial),
            ("corrected estimate", corrected, (decision,)),
        ),
        str(tmp_path / "decisions.html"),
    )

    html = destination.read_text(encoding="utf-8")
    assert '"relationship": "correction"' in html
    assert '"updated": true' in html
    assert '"old_fact_id": "volume-5000"' in html
    assert "Evidence &amp; reconciliation" in html
    assert "ArrowLeft" in html
    assert "ArrowRight" in html


def test_snapshot_timeline_bookmarks_policy_resolutions(tmp_path: Path) -> None:
    """Mark a snapshot whose evidence received policy-guided resolution."""
    graph = Graph(
        entities={"Incident", "Pier 7"},
        edges={"reported_at"},
        relations={("Incident", "reported_at", "Pier 7")},
    )
    decision = ReconciliationDecision(
        old_fact_id="pier-7",
        new_fact_id="pier-9",
        relationship="contradiction",
        rationale="Neither report has enough authority to win.",
        policy_name="disaster_mitigation_evidence_resolution",
        policy_version="4.0.0",
        policy_applied_for="contradiction",
        policy_resolution=EvidenceResolution(
            resolution=(
                "Neither report is independently corroborated; preserve both while "
                "direct field verification is pending."
            ),
        ),
    )

    destination = visualize_snapshots(
        (("initial", graph), ("disputed", graph, (decision,))),
        str(tmp_path / "policy-bookmark.html"),
    )

    html = destination.read_text(encoding="utf-8")
    assert '"policy_resolutions": [\n        "contradiction"' in html
    assert "policy-bookmark" in html
    assert "Policy resolution: ${escapeHtml(resolutionLabel)}" in html
    assert "Evidence handling" in html
    assert '"resolution": "Neither report is independently corroborated;' in html


def test_snapshot_timeline_preserves_raw_observation_json(tmp_path: Path) -> None:
    """Keep every source observation recoverable regardless of graph projection."""
    utc = timezone.utc
    first = Observation(
        fact_id="field-report-a",
        subject="Incident",
        relation="reported_at",
        object="Zone A",
        source_text="Field report retained verbatim, including </script> text.",
        valid_from=datetime(2026, 9, 4, 9, 0, tzinfo=utc),
        observed_at=datetime(2026, 9, 4, 9, 3, tzinfo=utc),
    )
    second = Observation(
        fact_id="field-report-b",
        subject="Incident",
        relation="reported_at",
        object="Zone B",
        source_text="A second team reported a different location.",
        valid_from=datetime(2026, 9, 4, 9, 0, tzinfo=utc),
        observed_at=datetime(2026, 9, 4, 9, 8, tzinfo=utc),
    )
    first_graph = Graph(
        entities={"Incident", "Zone A"},
        edges={"reported_at"},
        relations={("Incident", "reported_at", "Zone A")},
    )
    disputed_graph = Graph(entities=set(), edges=set(), relations=set())

    destination = visualize_snapshots(
        (
            ("09:03", first_graph, (), first),
            ("09:08", disputed_graph, (), second),
        ),
        str(tmp_path / "evidence-timeline.html"),
    )

    html = destination.read_text(encoding="utf-8")
    data_marker = '<script id="viz-data" type="application/json">\n'
    embedded_json = html.split(data_marker, 1)[1].split("</script>", 1)[0]
    payload = json.loads(embedded_json)

    assert [entry["snapshot_index"] for entry in payload["evidence_timeline"]] == [
        0,
        1,
    ]
    assert [entry["observation"] for entry in payload["evidence_timeline"]] == [
        first.model_dump(mode="json"),
        second.model_dump(mode="json"),
    ]
    assert payload["snapshots"][0]["evidence_indices"] == [0]
    assert payload["snapshots"][1]["evidence_indices"] == [1]
    assert "Raw Observation JSON" in html
    assert "<\\/script>" in embedded_json


def test_contradiction_presents_two_manual_override_options(tmp_path: Path) -> None:
    """Offer both raw observations for local selection without classification."""
    utc = timezone.utc
    old_observation = Observation(
        fact_id="zone-a-safe",
        subject="Zone A",
        relation="safety_status",
        object="Safe",
        source_text="Team One reported Zone A safe.",
        observed_at=datetime(2026, 9, 4, 9, 0, tzinfo=utc),
    )
    new_observation = Observation(
        fact_id="zone-a-unsafe",
        subject="Zone A",
        relation="safety_status",
        object="Unsafe",
        source_text="Team Two reported Zone A unsafe.",
        observed_at=datetime(2026, 9, 4, 9, 5, tzinfo=utc),
        valid_from=old_observation.observed_at,
    )
    decision = ReconciliationDecision(
        old_fact_id="zone-a-safe",
        new_fact_id="zone-a-unsafe",
        relationship="contradiction",
        rationale="The reports conflict for the same interval.",
    )
    initial_graph = Graph(
        entities={"Zone A", "Safe"},
        edges={"safety_status"},
        relations={("Zone A", "safety_status", "Safe")},
    )
    disputed_graph = Graph(entities=set(), edges=set(), relations=set())

    destination = visualize_snapshots(
        (
            ("09:00", initial_graph, (), old_observation),
            ("09:05", disputed_graph, (decision,), new_observation),
        ),
        str(tmp_path / "manual-override.html"),
    )

    html = destination.read_text(encoding="utf-8")
    data_marker = '<script id="viz-data" type="application/json">\n'
    embedded_json = html.split(data_marker, 1)[1].split("</script>", 1)[0]
    payload = json.loads(embedded_json)
    manual_override = payload["snapshots"][1]["manual_overrides"][0]

    assert manual_override["snapshot_index"] == 1
    assert [option["fact_id"] for option in manual_override["options"]] == [
        "zone-a-safe",
        "zone-a-unsafe",
    ]
    assert [
        payload["evidence_timeline"][option["evidence_index"]]["observation"]
        for option in manual_override["options"]
    ] == [
        old_observation.model_dump(mode="json"),
        new_observation.model_dump(mode="json"),
    ]
    assert "Manual contradiction override" in html
    assert "Pursue this evidence" in html
    assert "does not call an LLM" in html
    assert "Recompute snapshots from checkpoint" in html
    assert "manual-recompute-toggle" in html
