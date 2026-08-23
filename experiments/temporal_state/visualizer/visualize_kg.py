"""High fidelity visualization utilities for kg-gen knowledge graphs."""

from __future__ import annotations

import colorsys
import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable
import webbrowser

from kg_gen.models import Graph

from experiments.temporal_state.ledger import ResolutionPolicy
from experiments.temporal_state.models import (
    Observation,
    ReconciliationDecision,
    Relation,
    Relationship,
)


def _string_to_color(label: str) -> str:
    """Generate a deterministic pastel-like color for a given label."""
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    saturation = 0.55 + (int(digest[2:4], 16) / 255.0) * 0.3
    lightness = 0.45 + (int(digest[4:6], 16) / 255.0) * 0.25
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _sorted_ignore_case(items: Iterable[str]) -> list[str]:
    return sorted(items, key=lambda value: value.lower())


def _build_view_model(
    graph: Graph,
    *,
    updated_relations: set[Relation] | None = None,
) -> dict[str, Any]:
    updated_relations = updated_relations or set()
    # Collect all entities from both the entities set and relations
    all_entities = set(graph.entities)
    for subject, _, obj in graph.relations:
        all_entities.add(subject)
        all_entities.add(obj)
    entities = _sorted_ignore_case(all_entities)

    relations = sorted(
        graph.relations,
        key=lambda triple: (triple[1].lower(), triple[0].lower(), triple[2].lower()),
    )

    entity_clusters = graph.entity_clusters or {}
    edge_clusters = graph.edge_clusters or {}

    entity_member_to_cluster: dict[str, str] = {}
    cluster_view: list[dict[str, Any]] = []

    for representative, members in entity_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"entity::{representative}")
        cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            entity_member_to_cluster[member] = representative

    node_color_lookup: dict[str, str] = {}
    if cluster_view:
        for cluster in cluster_view:
            for member in cluster["members"]:
                node_color_lookup[member] = cluster["color"]
    else:
        for entity in entities:
            node_color_lookup[entity] = _string_to_color(f"entity::{entity}")

    edge_member_to_cluster: dict[str, str] = {}
    edge_color_lookup: dict[str, str] = {}
    edge_cluster_view: list[dict[str, Any]] = []

    for representative, members in edge_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"edge::{representative}")
        edge_cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            edge_member_to_cluster[member] = representative
            edge_color_lookup[member] = color

    degree = Counter()
    indegree = Counter()
    outdegree = Counter()
    predicate_counts = Counter()

    adjacency: dict[str, set[str]] = defaultdict(set)
    node_neighbors: dict[str, set[str]] = defaultdict(set)
    node_edges: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"incoming": [], "outgoing": []}
    )

    edges_view: list[dict[str, Any]] = []

    for index, (subject, predicate, obj) in enumerate(relations):
        predicate_counts[predicate] += 1
        degree[subject] += 1
        degree[obj] += 1
        outdegree[subject] += 1
        indegree[obj] += 1
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
        node_neighbors[subject].add(obj)
        node_neighbors[obj].add(subject)

        edge_id = f"e{index}"
        color = edge_color_lookup.get(predicate)
        if not color:
            color = _string_to_color(f"predicate::{predicate}")
            edge_color_lookup[predicate] = color

        edges_view.append(
            {
                "id": edge_id,
                "source": subject,
                "target": obj,
                "predicate": predicate,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": color,
                "tooltip": f"{subject} —{predicate}→ {obj}",
                "updated": (subject, predicate, obj) in updated_relations,
            }
        )

        node_edges[subject]["outgoing"].append(edge_id)
        node_edges[obj]["incoming"].append(edge_id)

    isolated_entities = [entity for entity in entities if degree[entity] == 0]

    def connected_components() -> list[dict[str, Any]]:
        visited: set[str] = set()
        components: list[dict[str, Any]] = []
        for node in entities:
            if node in visited:
                continue
            queue: deque[str] = deque([node])
            visited.add(node)
            members: list[str] = []
            while queue:
                current = queue.popleft()
                members.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(
                {
                    "size": len(members),
                    "members": _sorted_ignore_case(members),
                }
            )
        components.sort(key=lambda comp: (-comp["size"], comp["members"][0]))
        return components

    components = connected_components()

    nodes_view: list[dict[str, Any]] = []
    for entity in entities:
        cluster_id = entity_member_to_cluster.get(entity)
        radius = 18 + min(degree[entity], 8) * 2
        nodes_view.append(
            {
                "id": entity,
                "label": entity,
                "cluster": cluster_id,
                "color": node_color_lookup.get(entity, "#64748b"),
                "degree": degree[entity],
                "indegree": indegree[entity],
                "outdegree": outdegree[entity],
                "isRepresentative": cluster_id == entity if cluster_id else False,
                "radius": radius,
                "neighbors": _sorted_ignore_case(node_neighbors.get(entity, set())),
                "edgeIds": node_edges.get(entity, {"incoming": [], "outgoing": []}),
            }
        )

    top_entities = sorted(
        (
            {
                "label": node["label"],
                "degree": node["degree"],
                "indegree": node["indegree"],
                "outdegree": node["outdegree"],
                "cluster": node["cluster"],
            }
            for node in nodes_view
        ),
        key=lambda item: (-item["degree"], item["label"].lower()),
    )[:10]

    top_relations = sorted(
        (
            {
                "predicate": predicate,
                "count": count,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": edge_color_lookup.get(predicate, "#64748b"),
            }
            for predicate, count in predicate_counts.items()
        ),
        key=lambda item: (-item["count"], item["predicate"].lower()),
    )[:10]

    stats = {
        "entities": len(entities),
        "relations": len(edges_view),
        "relationTypes": len(predicate_counts),
        "entityClusters": len(cluster_view),
        "edgeClusters": len(edge_cluster_view),
        "isolatedEntities": len(isolated_entities),
        "components": len(components),
        "averageDegree": round(
            sum(degree[entity] for entity in entities) / len(entities), 2
        )
        if entities
        else 0,
        "density": round(len(edges_view) / (len(entities) * (len(entities) - 1)), 3)
        if len(entities) > 1
        else 0,
    }

    relation_records = [
        {
            "source": subject,
            "predicate": predicate,
            "target": obj,
            "edgeId": edge["id"],
            "color": edge["color"],
        }
        for edge, (subject, predicate, obj) in zip(edges_view, relations)
    ]

    return {
        "nodes": nodes_view,
        "edges": edges_view,
        "clusters": cluster_view,
        "edgeClusters": edge_cluster_view,
        "topEntities": top_entities,
        "topRelations": top_relations,
        "stats": stats,
        "isolatedEntities": isolated_entities,
        "components": components,
        "relations": relation_records,
    }


HTML_TEMPLATE = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")
COMPARISON_DASHBOARD_TEMPLATE = (
    Path(__file__).parent / "comparison_dashboard.html"
).read_text(encoding="utf-8")
OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output").resolve()


def open_output_visualization(
    filename: str | Path = "graph.html",
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Open an existing HTML visualization from the temporal-state output folder."""
    root = Path(output_dir).resolve() if output_dir is not None else OUTPUT_DIR
    destination = (root / filename).resolve()

    if destination.parent != root:
        raise ValueError("filename must be directly inside the output folder")
    if destination.suffix.lower() != ".html":
        raise ValueError("visualization filename must end in .html")
    if not destination.is_file():
        raise FileNotFoundError(f"visualization does not exist: {destination}")

    webbrowser.open(destination.as_uri())
    return destination


def _write_visualization(
    payload: dict[str, Any],
    output_path: str | None,
    *,
    open_in_browser: bool,
) -> Path:
    """Inject visualization data into the template and write the HTML file."""
    serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2).replace(
        "</", "<\\/"
    )
    html = HTML_TEMPLATE.replace(
        "<!--DATA-->",
        serialized_payload,
    ).replace("<!--DATA_CONFIG-->", "{}")

    # Make sidebar visible for standalone mode by removing display: none.
    html = html.replace(
        "display: none; /* Hidden by default - controlled by main app */",
        "display: block; /* Visible in standalone mode */",
    )

    destination = Path(output_path or "graph-visualization.html").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(destination.as_uri())

    return destination


def write_snapshot_viewer(
    datasets: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    default_version: str | None = None,
    open_in_browser: bool = False,
) -> Path:
    """Write one HTML viewer that loads a selected versioned JSON dataset."""
    if not datasets:
        raise ValueError("at least one snapshot dataset is required")

    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".html":
        raise ValueError("snapshot viewer output must end in .html")

    versions = tuple(datasets)
    selected_default = default_version or versions[0]
    if selected_default not in datasets:
        raise ValueError(f"unknown default dataset version: {selected_default}")

    dataset_entries = []
    for version, data_path in datasets.items():
        resolved_data_path = Path(data_path).resolve()
        if resolved_data_path.suffix.lower() != ".json":
            raise ValueError("snapshot dataset paths must end in .json")
        if resolved_data_path.parent != destination.parent:
            raise ValueError("snapshot datasets must be beside the HTML viewer")
        dataset_entries.append(
            {"version": version, "path": resolved_data_path.name}
        )

    data_config = {
        "datasets": dataset_entries,
        "default_version": selected_default,
    }
    serialized_config = json.dumps(
        data_config, ensure_ascii=False, indent=2
    ).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("<!--DATA-->", "{}").replace(
        "<!--DATA_CONFIG-->", serialized_config
    )
    html = html.replace(
        "display: none; /* Hidden by default - controlled by main app */",
        "display: block; /* Visible in standalone mode */",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    if open_in_browser:
        webbrowser.open(destination.as_uri())
    return destination


def write_policy_comparison_dashboard(
    output_path: str | Path,
    *,
    open_in_browser: bool = False,
) -> Path:
    """Write a dashboard for comparing up to four snapshot JSON datasets."""
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".html":
        raise ValueError("comparison dashboard output must end in .html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(COMPARISON_DASHBOARD_TEMPLATE, encoding="utf-8")
    if open_in_browser:
        webbrowser.open(destination.as_uri())
    return destination


def visualize(
    graph: Graph,
    output_path: str | None = None,
    *,
    open_in_browser: bool = False,
) -> Path:
    """Render an interactive dashboard for a graph.

    Args:
        graph: Graph instance to visualize.
        output_path: Optional path where the HTML document should be stored.
        open_in_browser: When True, open the generated file in the default browser.

    Returns:
        Path to the generated HTML file.
    """

    if not graph or not graph.entities:
        raise ValueError("Cannot visualize an empty graph")

    return _write_visualization(
        _build_view_model(graph),
        output_path,
        open_in_browser=open_in_browser,
    )


def _build_snapshot_payload(
    snapshots: Iterable[
        tuple[str, Graph]
        | tuple[str, Graph, Iterable[ReconciliationDecision]]
        | tuple[
            str,
            Graph,
            Iterable[ReconciliationDecision],
            Observation | Iterable[Observation],
        ]
    ],
    *,
    policy_only_snapshots: bool = False,
    policy: ResolutionPolicy | None = None,
) -> dict[str, Any]:
    """Build the serializable timeline payload for graph snapshots.

    Args:
        snapshots: Ordered ``(label, graph)``, ``(label, graph, decisions)``, or
            ``(label, graph, decisions, observations)`` tuples. The fourth value
            accepts one raw Observation or an iterable of them. Labels are displayed
            beside the timeline slider and should identify each time interval.
        policy_only_snapshots: Initial state of the HTML policy-event filter.
            All snapshots are still computed and embedded when this is True.
        policy: Resolution policy whose instruction keys qualify a snapshot for
            the policy-event filter. Empty instructions default to all relationships.

    Returns:
        JSON-serializable visualization data.
    """
    snapshot_list = list(snapshots)
    if not snapshot_list:
        raise ValueError("Cannot visualize an empty snapshot sequence")
    enabled_relationships = policy.policy_relationships if policy is not None else ()
    enabled_relationship_set = set(enabled_relationships)
    inferred_policy_relationships: list[Relationship] = []

    rendered_snapshots = []
    evidence_timeline = []
    evidence_index_by_fact_id: dict[str, int] = {}
    previous_relations: set[Relation] = set()
    for snapshot_index, snapshot in enumerate(snapshot_list):
        if len(snapshot) == 2:
            label, graph = snapshot
            decisions: Iterable[ReconciliationDecision] = ()
            observations: Observation | Iterable[Observation] = ()
        elif len(snapshot) == 3:
            label, graph, decisions = snapshot
            observations = ()
        elif len(snapshot) == 4:
            label, graph, decisions, observations = snapshot
        else:
            raise ValueError("snapshot tuples must contain 2, 3, or 4 values")

        observation_list = (
            [observations]
            if isinstance(observations, Observation)
            else list(observations)
        )
        if not all(
            isinstance(observation, Observation) for observation in observation_list
        ):
            raise TypeError("snapshot evidence must contain Observation instances")

        evidence_indices = []
        for observation in observation_list:
            evidence_index = len(evidence_timeline)
            evidence_indices.append(evidence_index)
            evidence_timeline.append(
                {
                    "snapshot_index": snapshot_index,
                    "snapshot_label": str(label),
                    "observation": observation.model_dump(mode="json"),
                }
            )
            if observation.fact_id is not None:
                evidence_index_by_fact_id[observation.fact_id] = evidence_index

        current_relations = set(graph.relations)
        added_relations = current_relations - previous_relations
        removed_relations = previous_relations - current_relations
        decision_list = list(decisions)
        epistemic_relationships = sorted(
            {decision.relationship for decision in decision_list}
        )
        policy_resolutions = sorted(
            {
                decision.policy_applied_for
                for decision in decision_list
                if decision.policy_applied_for is not None
            }
        )
        for relationship in policy_resolutions:
            if relationship not in inferred_policy_relationships:
                inferred_policy_relationships.append(relationship)
        manual_overrides = []
        for decision in decision_list:
            if decision.relationship != "contradiction":
                continue
            option_indices = [
                evidence_index_by_fact_id.get(decision.old_fact_id),
                evidence_index_by_fact_id.get(decision.new_fact_id),
            ]
            if any(index is None for index in option_indices):
                continue

            conflicting_relations = {
                (
                    evidence_timeline[evidence_index]["observation"]["subject"],
                    evidence_timeline[evidence_index]["observation"]["relation"],
                    evidence_timeline[evidence_index]["observation"]["object"],
                )
                for evidence_index in option_indices
                if evidence_index is not None
            }
            options = []
            for evidence_index in option_indices:
                if evidence_index is None:
                    continue
                observation = evidence_timeline[evidence_index]["observation"]
                relation = (
                    observation["subject"],
                    observation["relation"],
                    observation["object"],
                )
                override_relations = (
                    set(graph.relations) - conflicting_relations
                ) | {relation}
                override_graph = graph.model_copy(
                    update={
                        "entities": {
                            entity
                            for subject, _, object_ in override_relations
                            for entity in (subject, object_)
                        },
                        "edges": {
                            predicate for _, predicate, _ in override_relations
                        },
                        "relations": override_relations,
                    }
                )
                options.append(
                    {
                        "fact_id": observation["fact_id"],
                        "evidence_index": evidence_index,
                        "data": _build_view_model(
                            override_graph,
                            updated_relations={relation},
                        ),
                    }
                )
            manual_overrides.append(
                {
                    "decision_id": decision.decision_id,
                    "snapshot_index": snapshot_index,
                    "options": options,
                }
            )
        rendered_snapshots.append(
            {
                "label": str(label),
                "data": _build_view_model(
                    graph,
                    updated_relations=added_relations,
                ),
                "decisions": [
                    decision.model_dump(mode="json") for decision in decision_list
                ],
                "epistemic_relationships": epistemic_relationships,
                "policy_relevant": bool(
                    set(epistemic_relationships) & enabled_relationship_set
                    if policy is not None
                    else policy_resolutions
                ),
                "policy_resolutions": policy_resolutions,
                "evidence_indices": evidence_indices,
                "manual_overrides": manual_overrides,
                "changes": {
                    "added": [list(relation) for relation in sorted(added_relations)],
                    "removed": [
                        list(relation) for relation in sorted(removed_relations)
                    ],
                },
            }
        )
        previous_relations = current_relations

    return {
        "policy_definition": (
            policy.model_dump(mode="json") if policy is not None else None
        ),
        "snapshots": rendered_snapshots,
        "evidence_timeline": evidence_timeline,
        "timeline_config": {
            "policy_only_snapshots": policy_only_snapshots,
            "enabled_policy_relationships": list(
                enabled_relationships
                if policy is not None
                else inferred_policy_relationships
            ),
        },
    }


def write_snapshot_data(
    snapshots: Iterable[
        tuple[str, Graph]
        | tuple[str, Graph, Iterable[ReconciliationDecision]]
        | tuple[
            str,
            Graph,
            Iterable[ReconciliationDecision],
            Observation | Iterable[Observation],
        ]
    ],
    output_path: str | Path,
    *,
    policy_only_snapshots: bool = False,
    policy: ResolutionPolicy | None = None,
) -> Path:
    """Write graph snapshots and raw reconciliation output to a JSON dataset."""
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("snapshot data output must end in .json")
    payload = _build_snapshot_payload(
        snapshots,
        policy_only_snapshots=policy_only_snapshots,
        policy=policy,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def visualize_snapshots(
    snapshots: Iterable[
        tuple[str, Graph]
        | tuple[str, Graph, Iterable[ReconciliationDecision]]
        | tuple[
            str,
            Graph,
            Iterable[ReconciliationDecision],
            Observation | Iterable[Observation],
        ]
    ],
    output_path: str | None = None,
    *,
    open_in_browser: bool = False,
    policy_only_snapshots: bool = False,
    policy: ResolutionPolicy | None = None,
) -> Path:
    """Render graph snapshots as one self-contained HTML visualization."""
    payload = _build_snapshot_payload(
        snapshots,
        policy_only_snapshots=policy_only_snapshots,
        policy=policy,
    )
    return _write_visualization(
        payload,
        output_path,
        open_in_browser=open_in_browser,
    )
