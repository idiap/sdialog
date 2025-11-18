# -*- coding: utf-8 -*-
"""
Given a path to a collection of dialogues, this script first cluster all the utterances in the collection
and then convert each dialogue to a sequence of a "discrete trajectory" by replacing each utterances
with its corresponding cluster id.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import json
import torch
import logging
import numpy as np

from tqdm.auto import tqdm
from networkx import DiGraph
from tenacity import RetryError
from matplotlib import pyplot as plt
from typing import List, Union, Dict, Tuple
from simpleneighbors import SimpleNeighbors
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, to_tree


from .util import SentenceTransformerOpenAI, slugify, get_turn_text, init_gpt, get_cluster_label
from .build_graph import trajectory2graph
from .. import Dialog, config


DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
DEFAULT_SYS_NAME = "system"
DEFAULT_USER_NAME = "user"
DEFAULT_TOKEN_START = "[start]"
DEFAULT_TOKEN_END = "[end]"

seed = 13

speakers = [DEFAULT_SYS_NAME, DEFAULT_USER_NAME]
speaker2mame = {}
name2speaker = {}

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s')

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_turn_tag(speaker_name: str) -> str:
    """
    Returns the tag for the turn based on the speaker name.
    """
    speaker_name = speaker_name.lower()
    if speaker_name in speakers:
        return speaker_name

    if speaker_name not in name2speaker:
        name2speaker[speaker_name] = speakers[len(name2speaker) % len(speakers)]
        if name2speaker[speaker_name] not in speaker2mame:  # Added only once, we assume only two speakers
            speaker2mame[name2speaker[speaker_name]] = speaker_name

    return name2speaker[speaker_name]


def plot_dendrogram(model, title, path, labels=None, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    root, nodes = to_tree(linkage_matrix, rd=True)

    def get_leaves_of(node):
        # if it is a leaf node
        if node.count == 1 and node.dist == 0:
            return set([model.labels_[node.id]])
        return get_leaves_of(node.left).union(get_leaves_of(node.right))

    def get_children_leaf_ids(cluster_id):
        node = [node for node in nodes if node.id == cluster_id][0]
        return get_leaves_of(node)

    labeled = []

    def leaf2label(id):
        # if id < n_samples:
        if labels and model.labels_[id] not in labeled:
            labeled.append(model.labels_[id])
            return labels[model.labels_[id]]
        return str(model.labels_[id])

    def link_color_func(id):
        leaves_cluster_ids = get_children_leaf_ids(id)
        if len(leaves_cluster_ids) > 1:
            return "black"
        cluster_id = list(leaves_cluster_ids)[0]
        return f"C{cluster_id}"

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,
               leaf_label_func=leaf2label,
               link_color_func=link_color_func,
               no_labels=True,
               leaf_rotation=-90,
               **kwargs)

    ax = plt.gca()
    ax.set_ylim([0, .8])
    plt.ylabel('cosine distance', fontsize=12)
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.show()


def dialog2trajectories(
    input_dialogues: Union[str, List[Dialog]],
    system_speaker_name: str = None,
    output_path: str = None,
    embedding_model: str = "sergioburdisso/dialog2flow-joint-bert-base",
    thresholds: Union[Union[float, int], List[Union[float, int]]] = .6,  # [system threshold, user threshold]
    labels_enabled: bool = False,
    labels_model: str = None,
    labels_top_k: int = 5,
    dendrogram: bool = True,
    target_domains: List[str] = None,
    verbose: bool = True,
) -> str:
    global speaker2mame, name2speaker
    log_level = logging.INFO if verbose else logging.DEBUG

    if system_speaker_name:
        system_speaker_name = system_speaker_name.lower()
        name2speaker[system_speaker_name] = DEFAULT_SYS_NAME
        speaker2mame[DEFAULT_SYS_NAME] = system_speaker_name

    if labels_model is None:
        labels_model = config["llm"]["model"]

    if type(thresholds) is not list:
        thresholds = [thresholds]

    domain = "default"
    # TODO: remove all this code, use `input_dialogues = Dialog.from_file(path)` instead
    if isinstance(input_dialogues, str) and os.path.isdir(input_dialogues):
        logger.log(log_level, "Reading conversations...")
        if not os.path.exists(input_dialogues):
            raise FileNotFoundError(f"The provided input path is not a valid path: '{input_dialogues}'")
        if not output_path:
            output_path = os.path.join(input_dialogues, "dialog2flow")

        path_dialogues = input_dialogues
        input_dialogues = []
        domain = os.path.basename(os.path.normpath(path_dialogues))
        for filename in tqdm(os.listdir(path_dialogues), desc="Reading dialogues"):
            if os.path.isdir(os.path.join(path_dialogues, filename)):
                continue

            dialog_id, ext = os.path.splitext(filename)
            if ext in [".json", ".txt", "csv", "tsv"]:
                dialog = Dialog.from_file(os.path.join(path_dialogues, filename))
                if dialog.id is None:
                    dialog.id = dialog_id
                input_dialogues.append(dialog)
            else:
                logger.warning(f"File extension '{ext}' not supported: skipping file '{filename}'")
                continue

    if not output_path:
        output_path = ".dialog2flow/"
    dialogues = {}
    speaker2mame, name2speaker = {}, {}
    if isinstance(input_dialogues, list) and all(isinstance(d, Dialog) for d in input_dialogues):
        for dialog in input_dialogues:
            dialogue = [{"tag": get_turn_tag(turn.speaker),
                         "text": turn.text.strip(),
                         "turn": None} for turn in dialog.turns]
            dialogues[dialog.id] = {
                "goal": {domain: {}},
                "log": [
                    {
                        "tag": None,
                        "text": None,
                        "turn": DEFAULT_TOKEN_START
                    },
                    {
                        "tag": None,
                        "text": None,
                        "turn": DEFAULT_TOKEN_END
                    }
                ]
            }
            dialogues[dialog.id]["log"] = dialogues[dialog.id]["log"][:1] + dialogue + dialogues[dialog.id]["log"][-1:]
    else:
        raise ValueError("Input dialogues (`input_dialogues`) must be either a list of `Dialog` objects "
                         "or a path to the folder containing the dialogues")

    model_name = slugify(os.path.basename(embedding_model))
    output_path_trajectories = os.path.join(output_path, f"trajectories-{model_name}.json")
    output_path_clusters_folder = os.path.join(os.path.join(output_path, "clusters", model_name))

    domains = {}
    new_dialogs = {}
    unique_domains = set()
    for dialog_id, dialogue in dialogues.items():
        domain = next(iter(dialogue["goal"]))
        unique_domains.add(domain)

        if target_domains and domain not in target_domains:
            continue

        new_dialogs[dialog_id] = dialogue

        if domain not in domains:
            domains[domain] = {"log": [], "speaker": [], "text": [],
                               "emb": None, "prediction": None}
        domains[domain]["speaker"].extend(turn["tag"].lower() for turn in dialogue["log"][1:-1])
        domains[domain]["text"].extend(get_turn_text(turn) for turn in dialogue["log"][1:-1])
        domains[domain]["log"].extend(dialogue["log"][1:-1])

    multi_domain = len(unique_domains) > 1

    logger.log(log_level, f"Using model '{embedding_model}' model to generate the embeddings.")
    pb_domain = tqdm(domains, desc="Domains") if multi_domain else domains
    for domain in pb_domain:
        if multi_domain:
            logger.log(log_level, f"Domain: {domain.upper()}")

        domains[domain]["speaker"] = np.array(domains[domain]["speaker"])
        domains[domain]["text"] = np.array(domains[domain]["text"])
        domains[domain]["prediction"] = np.zeros_like(domains[domain]["text"], dtype=int)
        domains[domain]["labels"] = np.array([get_turn_text(t, use_ground_truth=True)
                                              for t in domains[domain]["log"]])

        if embedding_model.lower() == "chatgpt" or "openai" in embedding_model.lower():
            if "openai" in embedding_model.lower() and "/" in embedding_model:  # e.g. openai/text-embedding-3-large
                embedding_model = os.path.basename(embedding_model)
            else:
                embedding_model = DEFAULT_OPENAI_MODEL
            sentence_encoder = SentenceTransformerOpenAI(embedding_model)
        else:
            sentence_encoder = SentenceTransformer(embedding_model, device=device)

        domains[domain]["emb"] = sentence_encoder.encode(domains[domain]["text"],
                                                         show_progress_bar=True,  # show_progress_bar=verbose,
                                                         batch_size=128, device=device)
        # GloVe can return some Zero vectors, which invalidate the use of cosine distance, seting
        # one coordinate to 1 as a quick work around to prevent division by zero error:
        domains[domain]["emb"][np.where(~np.any(domains[domain]["emb"], axis=1))[0], 0] = 1

        normalized_turn_names = {DEFAULT_USER_NAME: {}, DEFAULT_SYS_NAME: {}}
        for spix, speaker in enumerate(sorted(normalized_turn_names.keys())):
            logger.log(log_level, f"Clustering {speaker.upper()} utterances...")
            speaker_mask = domains[domain]["speaker"] == speaker
            linkage = "average"
            n_clusters = None
            n_unique_labels = None
            distance_threshold = None

            if not speaker_mask.any():
                logger.warning(f"No {speaker} utterances were found.")
                continue

            threshold = thresholds[min(spix, len(thresholds) - 1)]  # system threshold, user threshold
            if threshold is None or threshold < 0:
                logger.log(log_level,
                           "No valid threshold or number of cluster was provided. "
                           "Trying to set the number of clusters using ground truth annotation (if available)")
                unique_labels = np.unique(domains[domain]["labels"][speaker_mask]).tolist()
                if unique_labels == ["unknown"]:
                    raise ValueError("No ground truth annotation found "
                                     "(and `thresholds` was not provided or is invalid).")

                n_unique_labels = len(unique_labels)
                n_clusters = n_unique_labels
            elif threshold > 1 and threshold == int(threshold):
                n_clusters = int(threshold)
            else:
                distance_threshold = threshold

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric="cosine",
                compute_distances=True,
                distance_threshold=distance_threshold
            ).fit(domains[domain]["emb"][speaker_mask])
            predictions = clustering.labels_

            # Getting utterance closer to the centroid
            cluster_ids = np.unique(predictions)
            cluster_topk_utts = [None] * cluster_ids.shape[0]
            centroids = np.zeros((cluster_ids.shape[0], domains[domain]["emb"][0].shape[0]))
            for ix, cluster_id in enumerate(cluster_ids):
                cluster_utts = domains[domain]["text"][speaker_mask][predictions == cluster_id]
                cluster_embs = domains[domain]["emb"][speaker_mask][predictions == cluster_id]

                index = SimpleNeighbors(domains[domain]["emb"].shape[1], metric="cosine")
                index.feed([(utt, cluster_embs[cix]) for cix, utt in enumerate(cluster_utts)])
                index.build()

                centroids[ix] = cluster_embs.mean(axis=0)
                top_k = labels_top_k
                while cluster_topk_utts[ix] is None and top_k > 0:
                    try:
                        cluster_topk_utts[ix] = {"name": None, "utterances": index.nearest(centroids[ix], top_k)}
                    except ValueError:  # "Expected n_neighbors <= n_samples_fit"
                        top_k -= 1

            # Saving cluster information for later use (centroid embeddings and top-k utterances of each cluster)
            if labels_enabled:
                try:
                    init_gpt(labels_model)
                    for cluster in tqdm(cluster_topk_utts, desc=f"Cluster labels ({speaker.title()}):"):
                        cluster["name"] = get_cluster_label(cluster["utterances"], labels_model)
                except RetryError:
                    error_details = ""
                    if "gpt" not in labels_model:
                        error_details = ("Is ollama server running (`ollama serve`)? "
                                         "is the model locally availbale (`ollama list`)?")
                    raise ValueError("Error while trying to generate node labels with LLM model "
                                     f"`{labels_model}`. {error_details}")

            if multi_domain:
                output_path_clusters = os.path.join(output_path_clusters_folder, domain)
            else:
                output_path_clusters = output_path_clusters_folder

            os.makedirs(output_path_clusters, exist_ok=True)
            with open(os.path.join(output_path_clusters, f"centroid-embeddings.{speaker.lower()}.npy"), "wb") as writer:
                np.save(writer, centroids)
            with open(os.path.join(output_path_clusters, f"top-utterances.{speaker.lower()}.json"), "w") as writer:
                json.dump(cluster_topk_utts, writer)

            logger.log(log_level, f"# clusters: {len(np.unique(predictions))}")
            logger.log(log_level, f"# ground truth labels: {n_unique_labels}")
            logger.log(log_level, f"# Total predictions: {len(predictions)}")
            domains[domain]["prediction"][speaker_mask] = predictions
            for tid in np.unique(predictions):
                if cluster_topk_utts[tid]["name"] is None:
                    cluster_name = cluster_topk_utts[tid]["utterances"][0]
                else:
                    cluster_name = cluster_topk_utts[tid]["name"]
                normalized_turn_names[speaker][tid] = {"name": f"{tid}_" + cluster_name,
                                                       "info": cluster_topk_utts[tid],
                                                       "id": f"{speaker[0].lower()}{tid}"}

            if dendrogram:
                plots_path = os.path.join(output_path, "plots")
                if multi_domain:
                    plots_path = os.path.join(plots_path, domain)
                os.makedirs(plots_path, exist_ok=True)
                output_file = os.path.join(plots_path, f"dendrogram_{model_name}.{speaker.lower()}.png")
                plot_dendrogram(clustering,
                                f"{speaker.title()} Utterances ({model_name})",
                                output_file)
                logger.log(log_level, f"Dendrogram plot for {speaker} utterances saved in `{output_file}`")

        if not domains[domain]['prediction'].any():
            logger.warning(f"No cluster predictions for '{domain}'. Skipped.")
            continue

        for ix, turn in enumerate(domains[domain]["log"]):
            turn["turn"] = normalized_turn_names[turn['tag']][domains[domain]['prediction'][ix]]

        # Saving dialogues as state sequences for graph visualization (as url hash #)
        state_sequences = {did: f'#{",".join([t["turn"]["id"] for t in d["log"][1:-1]])}'
                           for did, d in new_dialogs.items() if domain in d["goal"]}
        with open(os.path.join(output_path_clusters, "cluster-id-sequences.json"), "w") as writer:
            json.dump(state_sequences, writer)
        with open(os.path.join(output_path_clusters, "metadata.json"), "w") as writer:
            json.dump({"model": embedding_model,
                       "speakers": name2speaker}, writer)

        for ix, turn in enumerate(domains[domain]["log"]):
            name = normalized_turn_names[turn['tag']][domains[domain]['prediction'][ix]]['name']
            turn["turn"] = f"{turn['tag'].upper()}: {name}"

    os.makedirs(output_path, exist_ok=True)
    with open(output_path_trajectories, "w") as writer:
        json.dump(new_dialogs, writer)

    return output_path_trajectories


def dialog2graph(
    input_dialogues: Union[str, List[Dialog]],
    system_speaker_name: str = None,
    output_path: str = None,
    node_embedding_model: str = "sergioburdisso/dialog2flow-joint-bert-base",
    node_thresholds: Union[Union[float, int], List[Union[float, int]]] = .55,  # [system threshold, user threshold]
    nodes_prune_threshold: float = 0.023,
    node_llm_labels_enabled: bool = True,
    node_llm_labels_model: str = None,
    node_llm_labels_top_k: int = 5,
    node_show_ids: bool = True,
    edges_weight_type: str = "prob-out",
    edges_prune_threshold: float = 0.05,
    out_png: bool = True,
    out_interactive: bool = False,
    target_domains: List[str] = None,
    verbose: bool = True
) -> Tuple[DiGraph, Dict[str, Dict]]:

    path_trajectories = dialog2trajectories(
        input_dialogues=input_dialogues,
        system_speaker_name=system_speaker_name,
        output_path=output_path,
        embedding_model=node_embedding_model,
        thresholds=node_thresholds,
        labels_enabled=node_llm_labels_enabled,
        labels_model=node_llm_labels_model,
        labels_top_k=node_llm_labels_top_k,
        dendrogram=False,
        target_domains=target_domains,
        verbose=verbose
    )

    return trajectory2graph(
        path_trajectories=path_trajectories,
        output_folder=os.path.join(os.path.split(path_trajectories)[0], "graph"),
        edges_weight=edges_weight_type,
        prune_threshold_nodes=nodes_prune_threshold,
        prune_threshold_edges=edges_prune_threshold,
        png_show_ids=node_show_ids,
        png_visualization=out_png,
        interactive_visualization=out_interactive,
        verbose=verbose
    )
