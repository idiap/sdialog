# -*- coding: utf-8 -*-
"""
Given a collection of dialogue action trajectories, this script convert them into
a single action transition graph that represent them.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import json
import shutil
import logging
import numpy as np
import networkx as nx

from graphviz import Digraph
from typing import List, Dict, Tuple

from .util import CaselessDict


DEFAULT_SYS_NAME = "system"
DEFAULT_USER_NAME = "user"
DEFAULT_TOKEN_START = "[start]"
DEFAULT_TOKEN_END = "[end]"
NODE_UTTERANCE_LEN = 30

logger = logging.getLogger(__name__)


class WidestWeight:
    def __init__(self, weight, inverse=True):
        self._value = 1 / weight if inverse else weight

    def __add__(self, weight):
        weight = weight._value if isinstance(weight, WidestWeight) else weight
        return WidestWeight(max(self._value, weight), inverse=False)

    def __radd__(self, weight):
        return self.__add__(weight)

    def __lt__(self, weight):
        weight = weight._value if isinstance(weight, WidestWeight) else weight
        return self._value < weight

    @staticmethod
    def nx_weight(weight="weight"):
        return lambda u, v, data: WidestWeight(data[weight])


def node2turn(node):
    node = node.replace("system: ", "agent: ").capitalize()
    m = re.match(r"(.+):\s+(?:\d+_)?(.+)", node)
    return f"{m.group(1)}: {m.group(2).capitalize()}"


def get_utterance(node):
    m = re.match(r".+:\s+\d+_(.+)", node)
    return m.group(1).capitalize()


def get_speaker(node):
    return "system" if node.lower().startswith("system") else "user"


def get_node_id(node):
    return node.split("_")[0].lower().replace("[", "").replace("]", "").replace(" ", "_")


def get_node_name(node, label=False, no_cluster_ids=False, show_id=False):
    if label:
        if re.search(r"\d", node):
            node = node.replace("system: ", "S").replace("user: ", "U")
            m = re.match(r"([SU].+?)_(.+)", node)
            utterance = "<BR/>".join([m.group(2)[ix * NODE_UTTERANCE_LEN: (ix + 1) * NODE_UTTERANCE_LEN]
                                      for ix in range(len(m.group(2)) // NODE_UTTERANCE_LEN + 1)])
            if no_cluster_ids:
                return f'<{utterance.capitalize()}>' if not show_id else f"<{utterance}<B>[{m.group(1)}]</B>>"
            else:
                return f'<<B>{m.group(1)}</B><I>("{utterance}")</I>>'
        else:
            return "<<B>" + node.replace("system: ", "").replace("user: ", "").upper() + "</B>>"

    return node.replace("system: ", "S").replace("user: ", "U")


def get_tooltip(info, node_id):
    if not info or not node_id[1:].isdigit():
        return ""
    speaker = "system" if node_id[0] == "s" else "user"
    ix = int(node_id[1:])
    return "\n".join(f"- {utt}" for utt in info[speaker][ix]["utterances"][:3])


def prune_graph(G, threshold=0.023, by="node", remove_unrecheable=True):  # by= ["node", "edge", "both"]
    if by in ["node", "both"]:
        G.remove_nodes_from([n for n, weight in G.nodes(data='weight') if weight < threshold])
    if by in ["edge", "both"]:
        G.remove_edges_from([(u, v) for u, v, weight in G.edges(data='weight') if weight < threshold])
        G.remove_nodes_from(list(nx.isolates(G)))

    if remove_unrecheable:
        end2start_reachables = nx.ancestors(G, DEFAULT_TOKEN_END).intersection(nx.descendants(G, DEFAULT_TOKEN_START))
        G.remove_nodes_from([n for n in G.nodes()
                             if n not in end2start_reachables | {DEFAULT_TOKEN_START, DEFAULT_TOKEN_END}])


def normalize_edges(G, policy):  # policy= "max" or "max-out" or "sum-out"
    if policy == "max":
        max_fr = max([d["fr"] for _, _, d in G.edges(data=True)])
        for s0, s1, d in G.edges(data=True):
            d["weight"] = d["fr"] / max_fr
    elif "-out" in policy:
        fn = max if "max-" in policy else sum
        for node_id in G.nodes:
            out_edges = G.out_edges(node_id, data=True)
            if out_edges:
                total_fr = fn([d['fr'] for _, _, d in out_edges])
                for a, b, d in out_edges:
                    d['weight'] = d['fr'] / total_fr


def create_graph(trajectories: Dict,
                 output_folder: str,
                 clusters_info_folder: str = None,
                 edges_weight: str = "max-out",
                 prune_threshold_nodes: float = 0.023,
                 prune_threshold_edges: float = 0.2,
                 png_show_ids: bool = False,
                 png_visualization: bool = True,
                 interactive_visualization: bool = False,
                 verbose: bool = True) -> Tuple[nx.DiGraph, Dict[str, Dict]]:
    log_level = logging.INFO if verbose else logging.DEBUG

    G = nx.DiGraph()
    G.add_node(DEFAULT_TOKEN_START, color="green", fr=1)
    G.add_node(DEFAULT_TOKEN_END, color="gray", border_color='black', border_size=2, fr=1)

    node_info = {}
    cluster_centroids = {}
    nodes_are_labels = False
    if clusters_info_folder and os.path.exists(clusters_info_folder):
        for speaker in [DEFAULT_SYS_NAME, DEFAULT_USER_NAME]:
            with open(os.path.join(clusters_info_folder, f"top-utterances.{speaker}.json")) as reader:
                node_info[speaker] = json.load(reader)
            cluster_centroids[speaker] = np.load(os.path.join(clusters_info_folder,
                                                              f"centroid-embeddings.{speaker}.npy"))
        nodes_are_labels = node_info[speaker][0]["name"]
    with open(os.path.join(clusters_info_folder, "metadata.json")) as reader:
        metadata = json.load(reader)

    for trajectory in trajectories.values():
        for ix in range(len(trajectory) - 1):
            s0, s0_speaker, s0_acts = trajectory[ix]
            s1, s1_speaker, s1_acts = trajectory[ix + 1]

            # Skipping edges to/from "noise" clusters
            if (s0_acts and s0_acts.startswith("-1")) or (s1_acts and s1_acts.startswith("-1")):
                if s1_acts and not s1_acts.startswith("-1"):
                    if s1 not in G.nodes:
                        G.add_node(s1, fr=0)
                    G.nodes[s1]["fr"] += 1
                continue

            edge = G.get_edge_data(s0, s1)
            if not edge:
                G.add_edge(s0, s1,
                           fr=1,
                           label=s0_acts if s0_acts else "ε",
                           color="blue" if s1_speaker == DEFAULT_USER_NAME else "red")
            else:
                edge["fr"] += 1

            if s1_speaker and s1 != DEFAULT_TOKEN_END:
                if "fr" not in G.nodes[s1]:
                    G.nodes[s1]["fr"] = 0
                G.nodes[s1]["fr"] += 1

            if s0_speaker:
                G.nodes[s0]["color"] = "blue" if s0_speaker == DEFAULT_USER_NAME else "red"
                G.nodes[s0]["speaker"] = s0_speaker

    # Merge nodes with the same label
    if nodes_are_labels:
        label2nodes = {}
        for n in G.nodes:
            # TODO: instead of consider nodes duplicate if have the exact same label
            #       perhaps a similarity metric and a threshold can be used
            label = f"{get_speaker(n)}-{get_node_name(n, label=True, no_cluster_ids=True)}"
            if label not in label2nodes:
                label2nodes[label] = []
            label2nodes[label].append(n)

        # if repeated labels
        repeated_nodes = [nodes for nodes in label2nodes.values() if len(nodes) > 1]
        del label2nodes
        if repeated_nodes:
            logger.log(log_level, f"Found {len(repeated_nodes)} unique labels with repeated nodes to marge")
            logger.log(log_level, f"    > Number of nodes before mergin duplicates: {len(G.nodes)}")
            for nodes in repeated_nodes:
                node_original, node_duplicates = nodes[0], nodes[1:]

                # 1) Updating the in-bound edges to link to original only
                for s, _, data in G.in_edges(node_duplicates, data=True):
                    if G.has_edge(s, node_original):
                        G[s][node_original]["fr"] += data["fr"]
                    else:
                        G.add_edge(s, node_original, **data)

                # 2) Updating the out-bound edges to link to original only
                for _, t, data in G.out_edges(node_duplicates, data=True):
                    if G.has_edge(node_original, t):
                        G[node_original][t]["fr"] += data["fr"]
                    else:
                        G.add_edge(node_original, t, **data)

                # 3) Updating original node frequencies
                for n in node_duplicates:
                    G.nodes[node_original]["fr"] += G.nodes[n]["fr"]

                G.remove_nodes_from(node_duplicates)
    logger.log(log_level, f"    > Number of nodes after mergin duplicates: {len(G.nodes)}")

    # Normalize nodes
    max_fr = max([fr for _, fr in G.nodes(data="fr")])
    for node, d in G.nodes(data=True):
        if node in [DEFAULT_TOKEN_START, DEFAULT_TOKEN_END]:
            d["weight"] = 1
        else:
            d["weight"] = d["fr"] / max_fr

    normalize_edges(G, policy=edges_weight)

    logger.log(log_level, f"  #Nodes before pruning: {len(G.nodes)}")
    G.remove_edges_from(nx.selfloop_edges(G))
    prune_graph(G, threshold=prune_threshold_nodes)

    # Widest path ("Happy path")
    G2 = G.copy()
    edges_to_remove = []
    for s, t in G2.edges():
        if (s.startswith("user:") and t.startswith("user:")) or (s.startswith("system:") and t.startswith("system:")):
            edges_to_remove.append((s, t))
    G2.remove_edges_from(edges_to_remove)
    widest_path = nx.shortest_path(G2, DEFAULT_TOKEN_START, DEFAULT_TOKEN_END, weight=WidestWeight.nx_weight())
    with open(os.path.join(output_folder, "widest_path.txt"), "w") as writer:
        happy_path = [node2turn(n) for n in widest_path[1:-1]]
        logger.log(log_level, f"    Widest path: {happy_path}")
        writer.write("\n".join(happy_path))
    widest_path = [get_node_id(get_node_name(n)) for n in widest_path]  # for Javascript's `graph_happy_path`

    output_file = os.path.join(output_folder, "graph")
    g = Digraph('G', filename=output_file)
    g.node_attr.update(shape="underline", style="filled", fillcolor="white")

    prune_graph(G, prune_threshold_edges,
                by="edge",
                remove_unrecheable=True)
    logger.info(f"Final graph contains {len(G.nodes)} nodes after pruning and normalization.")

    normalize_edges(G, policy=edges_weight)  # normalizing again to recompute the weights

    for s0, s1, w in G.edges(data="weight"):
        try:
            color = None
            if "speaker" in G.nodes[s1]:
                color = "#0288d1" if G.nodes[s1]["speaker"] == DEFAULT_USER_NAME else "#9e9e9e"
            else:
                color = "#0288d1" if G.nodes[s0]["speaker"] == DEFAULT_USER_NAME else "#9e9e9e"
            g.edge(get_node_name(s0), get_node_name(s1),
                   penwidth=str(w * 5), color=color)
        except KeyError:
            g.edge(get_node_name(s0), get_node_name(s1), penwidth=str(w * 5))

    for n, data in G.nodes(data=True):
        if "speaker" in data:
            weight, speaker = data["weight"], data["speaker"]
            g.node(get_node_name(n),
                   label=get_node_name(n, label=True, no_cluster_ids=nodes_are_labels, show_id=png_show_ids),
                   penwidth=str(1 + weight * 5),
                   fillcolor="#b3e5fc" if speaker == DEFAULT_USER_NAME else "white")

    g.node(DEFAULT_TOKEN_START, "START", shape='Mdiamond', fillcolor="#e0e0e0")
    g.node(DEFAULT_TOKEN_END, "END", shape='Mdiamond', fillcolor="#e0e0e0")

    g.graph_attr["dpi"] = "300"
    # logger.log(log_level, f"  Saving graph as DOT format in '{output_file}.dot'")
    # g.render(output_file, view=False, format="dot")
    if png_visualization:
        logger.log(log_level, f"  Saving graph PNG visualization in '{output_file}.png'")
        g.render(output_file, view=False, format="png")
        try:
            from PIL import Image
            image = Image.open(f"{output_file}.png")
            image.show()
        except Exception:
            pass

    if interactive_visualization:
        output_folder = os.path.join(output_folder, "visualization")
        output_file = os.path.join(output_folder, "graph.html")

        logger.log(log_level, f"  Saving graph HTML interactive visualization in '{output_file}'")
        path_visualization = os.path.join(os.path.dirname(__file__), "util/visualization/")
        shutil.copytree(path_visualization, output_folder, dirs_exist_ok=True)
        with open(os.path.join(path_visualization, "graph.html")) as reader:
            html = reader.read()
        html_first, html_end = html.split("// [GRAPH HERE]")

        widest_path[:] = [f"'{node_id}'" for node_id in widest_path]
        graph_html = f"graph_happy_path = [{', '.join(widest_path)}]; "
        tooltips = {}
        for n, data in G.nodes(data=True):
            nid = get_node_id(get_node_name(n))
            nname = re.sub("<BR/>", "",
                           get_node_name(n, label=True, no_cluster_ids=nodes_are_labels).replace("'", r"\'")[1:-1],
                           flags=re.IGNORECASE)
            tooltips[nid] = get_tooltip(node_info, nid)
            if nid == "start":
                graph_html += (f"var v{nid} = graph.insertVertex(parent, '{nid}', '\t', 0, 0, 40, 10, "
                               "'fillColor=#B3B3B3;strokeColor=#03071e;")
            elif nid == "end":
                graph_html += f"var v{nid} = graph.insertVertex(parent, '{nid}', 'END', 0, 0, 50, 10, 'whiteSpace=wrap;"
            else:
                graph_html += (f"var v{nid} = graph.insertVertex(parent, '{nid}', '{nname}', 0, 0, 150, 10, "
                               "'whiteSpace=wrap;")

            if "speaker" in data:
                weight, speaker = data["weight"], data["speaker"]
                graph_html += f"strokeOpacity={weight * 100};fillColor={'#DC2F02' if speaker == 'user' else '#03071E'};"
            else:
                if nid == 'start':
                    graph_html += "shape=ellipse;fillColor=#B3B3B3;"
                else:
                    graph_html += "shape=ellipse;fillColor=#FFA500;"

            graph_html += "');"

        for eix, (s0, s1, w) in enumerate(G.edges(data="weight")):
            nname0, nname1 = get_node_name(s0), get_node_name(s1)
            nid0, nid1 = get_node_id(nname0), get_node_id(nname1)
            graph_html += (f"var e{eix} = graph.insertEdge(parent, null, '{w:.1%}', v{nid0}, v{nid1},"
                           "'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                           f"html=1;curved=1;endArrow=blockThin;endFill=1;strokeWidth={w * 4};")

            try:
                color = None
                if "speaker" in G.nodes[s1]:
                    color = "#3333AA" if G.nodes[s1]["speaker"] == DEFAULT_USER_NAME else "#cf8602"
                else:
                    color = "#3333AA" if G.nodes[s0]["speaker"] == DEFAULT_USER_NAME else "#cf8602"
                graph_html += f"strokeColor={color};"
            except KeyError:
                pass

            graph_html += "');"

        graph_html += f"tooltips = {json.dumps(tooltips)};"

        with open(output_file, "w") as writer:
            writer.write(html_first + graph_html + html_end)

    for speaker in node_info:
        for ix, info in enumerate(node_info[speaker]):
            info["centroid-embedding"] = cluster_centroids[speaker][ix]

    node2id_mapping = {}
    for node in G.nodes():
        if node.startswith("["):
            continue
        node_id = node.split("_")[0].lower().replace("system: ", "s").replace("user: ", "u")
        speaker, ix = node_id[0], int(node_id[1:])
        node2id_mapping[node] = node_id
    G = nx.relabel_nodes(G, node2id_mapping, copy=False)

    output_path = os.path.join(output_folder, "graph.graphml")
    logger.log(log_level, f"  Saving graph as GraphML format in '{output_path}'")
    nx.write_graphml(G, output_path)

    # Returning the graph and nodes info
    node_info = CaselessDict({f"{speaker[0].lower()}{ix}": info
                              for speaker in node_info for ix, info in enumerate(node_info[speaker])})
    if not metadata["speakers"]:
        metadata["speakers"] = {DEFAULT_SYS_NAME.lower(): DEFAULT_SYS_NAME,
                                DEFAULT_USER_NAME.lower(): DEFAULT_USER_NAME}
    metadata["output_folder"] = output_folder
    node_info["_metadata"] = metadata
    if png_visualization:
        node_info["_metadata"]["png"] = f"{output_file}.png"
    return G, node_info


def trajectory2graph(path_trajectories: str,
                     output_folder: str,
                     edges_weight: str = "prob-out",
                     prune_threshold_nodes: float = 0.023,
                     prune_threshold_edges: float = 0.2,
                     png_show_ids: bool = True,
                     png_visualization: bool = True,
                     interactive_visualization: bool = False,
                     target_domains: List[str] = None,
                     verbose: bool = True) -> Tuple[nx.DiGraph, Dict[str, Dict]]:
    log_level = logging.INFO if verbose else logging.DEBUG
    logger.log(log_level, f"  Reading trajectories from ({path_trajectories})...")
    with open(path_trajectories) as reader:
        data = json.load(reader)

    unique_domains = set()
    for dialog_id, dialogue in data.items():
        domain = next(iter(dialogue["goal"]))
        unique_domains.add(domain)
    multi_domain = len(unique_domains) > 1

    all_trajectories = {}
    for dialog_id in data:
        domain = next(iter(data[dialog_id]["goal"]))
        if target_domains and domain not in target_domains:
            continue

        if domain not in all_trajectories:
            all_trajectories[domain] = {}
        trajectories = all_trajectories[domain]
        trajectories[dialog_id] = []
        n_turns = len(data[dialog_id]["log"])
        for ix, turn in enumerate(data[dialog_id]["log"]):
            turn = turn["turn"]
            if ix == 0:
                trajectories[dialog_id].append((turn, None, None))
            elif ix >= n_turns - 1:
                trajectories[dialog_id].append((DEFAULT_TOKEN_END, None, None))
            else:
                # (id, speaker, acts)
                spkr_end_ix = turn.index(":")
                spkr, dial_act = turn[:spkr_end_ix], turn[spkr_end_ix + 1:].strip().replace(":", "")
                if re.match(r"^[a-z]\w*-(\w)", dial_act, flags=re.IGNORECASE):
                    domain, dial_act = dial_act.split("-")
                trajectories[dialog_id].append((f"{spkr.lower()}: {dial_act}", spkr.lower(), dial_act))

    for domain in all_trajectories:
        trajectories = all_trajectories[domain]
        logger.log(log_level,
                   f"    {len(trajectories)} trajectories read" + (f" for domain '{domain}'." if multi_domain else "."))

    for domain in all_trajectories:
        if multi_domain:
            logger.log(log_level, f"> Graph for domain: '{domain.upper()}'")
        logger.log(log_level, "  About to start creating the graph...")
        m = re.match(r".+trajectories-(.*).json", path_trajectories)
        model_name = m.group(1) if m else ""
        output_path = os.path.join(output_folder, model_name) if model_name else output_folder
        output_path = os.path.join(output_path, domain) if multi_domain else output_path
        os.makedirs(output_path, exist_ok=True)
        if model_name:
            output_path_clusters = os.path.join(os.path.split(path_trajectories)[0], "clusters", model_name)
            output_path_clusters = os.path.join(output_path_clusters, domain) if multi_domain else output_path_clusters
        else:
            output_path_clusters = None

        graph, nodes = create_graph(
            all_trajectories[domain],
            output_path,
            output_path_clusters,
            edges_weight,
            prune_threshold_nodes,
            prune_threshold_edges,
            png_show_ids,
            png_visualization,
            interactive_visualization,
            verbose
        )

        logger.log(log_level, "  Finished creating the graph.")

    return graph, nodes
