"""Regression checks for temporal graph visualization."""

from pathlib import Path
from unittest.mock import patch

from kg_gen.models import Graph

from experiments.temporal_state.models import ReconciliationDecision
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
    assert "Reconciliation" in html
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
