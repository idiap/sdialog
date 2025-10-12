"""
Utility Functions for sdialog

This module provides helper functions for the sdialog package, including serialization utilities to ensure
objects can be safely converted to JSON for storage or transmission.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import json
import uuid
import torch
import random
import logging
import subprocess
import numpy as np
import transformers
import pandas as pd

from time import sleep
from tqdm.auto import tqdm
from functools import wraps
from pydantic import BaseModel
from typing import Union, List, Tuple
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers.util import get_device_name, batch_to_device

from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables.base import RunnableBinding
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)

__version__ = "0.3.2"


def _get_dynamic_version() -> str:
    """ Retrieves the current version of the package, appending the current git commit hash if available."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        # If not a valid commit hash, set to empty string
        if re.match(r"\b[0-9a-f]{5,40}\b", commit_hash):
            return f"{__version__}+{commit_hash}"
    except Exception:
        pass
    return __version__


def dialogs_to_utt_pairs(dialogs: List[BaseModel], ai_speaker: str = None) -> Tuple[List[str], List[str]]:
    """
    Extract utterance -> next utterance (adjacent turn) pairs from dialogs.

    Two modes:
      * Sliding window mode (ai_speaker is None): pairs every turn with its successor.
      * QA mode (ai_speaker given): pairs each human turn immediately preceding an AI turn (speaker match).

    :param dialogs: List of dialog-like Pydantic objects each having a .turns list with .text and .speaker.
    :type dialogs: List[BaseModel]
    :param ai_speaker: Optional name (case-insensitive) of the AI speaker to filter answer turns.
    :type ai_speaker: str
    :return: Tuple (utterances, next_utterances) of equal length.
    :rtype: Tuple[List[str], List[str]]
    :raises ValueError: If no turns found, or lengths mismatch, or ai_speaker filtering yields nothing.
    """
    ai_speaker = ai_speaker.lower() if ai_speaker else None
    utts = []
    utts_next = []
    for dialog in dialogs:
        # if AI speaker is not specified, just return a sliding window of turns
        if not ai_speaker:
            turns = [t.text for t in dialog.turns]
            utts.extend(turns[:-1])
            utts_next.extend(turns[1:])
        else:  # If AI speaker is specified, return as human question and AI answer pairs
            ai_turns = [(ix, t.text)
                        for ix, t in enumerate(dialog.turns)
                        if t.speaker.lower() == ai_speaker]
            if not ai_turns:
                logger.warning(f"No turns found for AI speaker '{ai_speaker}' in dialog "
                               f"{dialog._path if hasattr(dialog, '_path') and dialog._path else ''}")
                continue

            for ix, _ in ai_turns:
                # Find the previous human turn (if exists)
                if ix > 0 and dialog.turns[ix - 1].speaker.lower() != ai_speaker:
                    utts.append(dialog.turns[ix - 1].text)
                    utts_next.append(dialog.turns[ix].text)

    if not utts or not utts_next:
        if ai_speaker:
            raise ValueError("No utterances found in the dialogs. Ensure the provided "
                             f"AI speaker ('{ai_speaker}') is correctly specified.")
        raise ValueError("No utterances found in the dialogs. Ensure the dialogs contain valid turns.")

    if len(utts) != len(utts_next):
        raise ValueError(f"Number of utterances ({len(utts)}) and next utterances ({len(utts_next)}) must be equal.")

    return utts, utts_next


def check_valid_model_name(func):
    """
    Decorator ensuring first argument (model_name) is a string; otherwise short-circuits to False.

    :param func: The predicate function to wrap.
    :type func: callable
    :return: Wrapped function enforcing a str model_name.
    :rtype: callable
    """
    def wrapper(model_name, *args, **kwargs):
        if not isinstance(model_name, str):
            return False
        return func(model_name, *args, **kwargs)
    return wrapper


def softmax(values, temperature=0.05, as_list=True):
    """
    Compute softmax over a 1D iterable of numeric values.

    :param values: Sequence of numeric scores.
    :type values: Iterable[float]
    :param temperature: Temperature divisor (lower = sharper distribution).
    :type temperature: float
    :param as_list: If True return a Python list; otherwise a torch tensor.
    :type as_list: bool
    :return: Softmax probability distribution.
    :rtype: Union[List[float], torch.Tensor]
    """
    probs = torch.nn.functional.softmax(torch.tensor(values, dtype=float) / temperature, dim=0)
    return probs.tolist() if as_list else probs


def get_universal_id() -> str:
    """
    Generates a unique identifier (UUID4) for sdialog objects.

    :return: A unique identifier as a string.
    :rtype: str
    """
    return str(uuid.uuid4())


def remove_newlines(s: str) -> str:
    """
    Replace all whitespace (including newlines) with single spaces and collapse repeats.

    :param s: Input value (non-str inputs are returned unchanged).
    :type s: Any
    :return: Normalized single-line string or original object if not str.
    :rtype: str
    """
    if type(s) is not str:
        return s
    return re.sub(r'\s+', ' ', s)


def get_timestamp() -> str:
    """
    Return current UTC timestamp in ISO 8601 format (seconds precision, trailing 'Z').

    :return: ISO 8601 UTC timestamp.
    :rtype: str
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def get_llm_default_params(model_name: str, llm_params: dict) -> float:
    """
    Get the default parameters for the model if not already specified, and merges them into `llm_params`.

    :param model_name: LLM model name.
    :type model_name: str
    :param llm_params: Existing LLM parameter dictionary to update in-place.
    :type llm_params: dict
    :return: Updated llm_params with defaults filled.
    :rtype: dict
    """
    if not is_ollama_model_name(model_name):
        return llm_params
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[-1]

    defaults = {}
    try:
        result = subprocess.run(
            ["ollama", "show", "--parameters", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        # Look for a line like: "temperature: 0.7"
        for line in result.stdout.splitlines():
            m = re.match(r'(\w+)\s+([0-9]*\.?[0-9]+)', line)  # For now only with numbers
            # TODO: add support for parsing string values later, gives ValidationError (probably the stop tokens?)
            # m = re.match(r'(\w+)\s+(.+)', line)
            if m:
                param, value = m.groups()
                if value.startswith('"'):
                    if param not in defaults:
                        defaults[param] = value.strip('"')
                    else:
                        if type(defaults[param]) is not list:
                            defaults[param] = [defaults[param]]
                        defaults[param].append(value.strip('"'))
                else:
                    try:
                        defaults[param] = float(value) if "." in value else int(value)
                    except ValueError:
                        logger.warning(f"Could not convert value '{value}' for parameter '{param}' "
                                       "to float or int. Skipping...")
        if "temperature" not in defaults:
            defaults["temperature"] = 0.8
    except Exception as e:
        logger.error(f"Error getting default parameters for model '{model_name}': {e}. Is Ollama server running?")
        logger.info("Trying to run the Ollama server...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sleep(5)  # Give it some time to start
            logger.info("Ollama server started. Retrying the request.")
            return get_llm_default_params(model_name, llm_params)
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")

    for k, v in list(defaults.items()):
        if k in llm_params and llm_params[k] is not None:
            continue
        llm_params[k] = v
    return llm_params


@check_valid_model_name
def is_ollama_model_name(model_name: str) -> bool:
    """
    Determine if model name refers to an Ollama model (excludes known OpenAI / HF / Google / AWS patterns).

    :param model_name: Model name string.
    :type model_name: str
    :return: True if considered an Ollama model.
    :rtype: bool
    """
    return (
        model_name.startswith("ollama:")
        or not is_huggingface_model_name(model_name)
        and not is_openai_model_name(model_name)
        and not is_google_genai_model_name(model_name)
        and not is_amazon_model_name(model_name)
    )


@check_valid_model_name
def is_openai_model_name(model_name: str) -> bool:
    """
    Check whether the model name targets an OpenAI chat model (prefix 'openai:').

    :param model_name: Model name string.
    :type model_name: str
    :return: True if OpenAI.
    :rtype: bool
    """
    return model_name.startswith("openai:")


@check_valid_model_name
def is_amazon_model_name(model_name: str) -> bool:
    """
    Check whether the model name targets an Amazon Bedrock model (prefix 'aws:' or 'amazon:').

    :param model_name: Model name string.
    :type model_name: str
    :return: True if Amazon Bedrock.
    :rtype: bool
    """
    return model_name.startswith("aws:") or model_name.startswith("amazon:")


@check_valid_model_name
def is_google_genai_model_name(model_name: str) -> bool:
    """
    Check whether the model name targets a Google Generative AI model (prefix 'google:' or 'google-genai:').

    :param model_name: Model name string.
    :type model_name: str
    :return: True if Google GenAI.
    :rtype: bool
    """
    return re.match(r"^google(-genai)?:", model_name, re.IGNORECASE)


@check_valid_model_name
def is_huggingface_model_name(model_name: str) -> bool:
    """
    Determine if a model name refers to a Hugging Face model (contains '/' or 'huggingface:' prefix).

    :param model_name: Model identifier.
    :type model_name: str
    :return: True if Hugging Face model.
    :rtype: bool
    """
    return model_name.startswith("huggingface:") or "/" in model_name


def get_llm_model(model_name: str,
                  output_format: Union[dict, BaseModel] = None,
                  return_model_params: bool = False,
                  think: bool = False,
                  tools: List = None,
                  **llm_kwargs):
    """
    Instantiate a LangChain chat model (OpenAI, AWS, Google, Ollama, Hugging Face).

    Applies backend-specific adjustments (e.g., removing unsupported params). Optionally
    wraps model for structured output if output_format provided and supported.

    :param model_name: Model name or instance.
    :type model_name: Union[str, Any]
    :param output_format: Pydantic model class or JSON schema dict for structured output.
    :type output_format: Union[dict, BaseModel, type[BaseModel]]
    :param return_model_params: If True, return (model, llm_kwargs) tuple instead of just model.
    :type return_model_params: bool
    :param think: If True, enables "thinking" segments in responses.
    :type think: bool
    :param tools: Optional list of tool functions to enable.
    :type tools: List[langchain_core.tools.structured.StructuredTool]
    :param llm_kwargs: Additional backend-specific model kwargs.
    :type llm_kwargs: dict
    :return: Configured LangChain model (possibly wrapped for structured output).
    :rtype: Any
    :raises ValueError: If model_name is invalid type.
    """
    # If model name has a slash, assume it's a Hugging Face model
    # Otherwise, assume it's an Ollama model
    llm_kwargs = get_llm_default_params(model_name, llm_kwargs)

    if not isinstance(model_name, str):
        if hasattr(model_name, "invoke") and callable(model_name.invoke):
            llm = model_name
        else:
            raise ValueError("model_name must be a string or a valid Langchain model instance.")
    elif is_openai_model_name(model_name):
        # If the model name is a string, assume it's an OpenAI model
        from langchain_openai import ChatOpenAI
        if ":" in model_name:
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading OpenAI model: {model_name}")

        llm = ChatOpenAI(model=model_name, **llm_kwargs)
    elif is_amazon_model_name(model_name):
        from langchain_aws import ChatBedrockConverse
        if ":" in model_name:
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading AWS model: {model_name}")

        if "seed" in llm_kwargs:
            logger.warning("Ignoring 'seed' parameter for AWS Bedrock models, as it is not supported.")
            llm_kwargs.pop("seed")

        llm = ChatBedrockConverse(model=model_name, **llm_kwargs)
    elif is_google_genai_model_name(model_name):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if ":" in model_name:
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading Google GenAI model: {model_name}")

        llm = ChatGoogleGenerativeAI(model=model_name, **llm_kwargs)
    elif is_ollama_model_name(model_name):
        if model_name.startswith("ollama:"):
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading Ollama model: {model_name}")

        ollama_check_and_pull_model(model_name)  # Ensure the model is available locally
        llm = ChatOllama(model=model_name, reasoning=think, **llm_kwargs)
    else:
        if model_name.startswith("huggingface:"):
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading Hugging Face model: {model_name}")
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        # Remove 'seed' from llm_kwargs if present (not supported by HuggingFace pipeline)
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if k != "seed"}
        llm_kwargs["model"] = model_name

        # Default HuggingFace parameters
        hf_defaults = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=2048,
            do_sample=True,
            repetition_penalty=1.03,
            return_full_text=False,
        )
        hf_params = {**hf_defaults, **llm_kwargs}

        pipe = transformers.pipeline("text-generation", **hf_params)

        llm = ChatHuggingFace(
            llm=HuggingFacePipeline(pipeline=pipe),
            tokenizer=AutoTokenizer.from_pretrained(model_name))  # if None, error (https://huggingface.co/models/None)

    if output_format:
        if isinstance(output_format, type) and issubclass(output_format, BaseModel):
            output_format = output_format.model_json_schema()
        if hasattr(llm, "with_structured_output"):
            llm = llm.with_structured_output(output_format)
        else:
            logger.error(f"The given model '{model_name}' does not support structured output. ")

    if tools:
        llm = llm.bind_tools(tools)

    return llm if not return_model_params else (llm, llm_kwargs)


def set_generator_seed(generator, seed):
    """
    Attempt to set a deterministic seed on the underlying LLM (if supported); fallback to torch.manual_seed.

    Also applies a workaround for certain Ollama caching issues by forcing an initial trivial generation.

    :param generator: Object containing .llm and (optionally) .messages or .memory.
    :type generator: Any
    :param seed: Desired seed; if None a random 32-bit value is generated.
    :type seed: int
    :return: The seed actually used (or None if unsupported).
    :rtype: int
    """
    seed = seed if seed is not None else random.getrandbits(32)
    llm = generator.llm
    base_model = None
    try:
        if isinstance(llm, BaseLanguageModel):
            base_model = llm
        elif isinstance(llm, RunnableBinding):
            base_model = llm.bound
        else:
            base_model = llm.steps[0].bound
        base_model.seed = seed
        logger.log(logging.DEBUG, f"Setting the LLM seed to {seed}...")
    except Exception:
        torch.manual_seed(seed)
        seed = None
        logger.warning("The LLM does not support dynamically setting a seed.")

    if isinstance(base_model, ChatOllama):
        # hack to avoid seed bug in prompt cache in Ollama
        # (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
        if hasattr(generator, "messages"):
            messages = generator.messages
        else:
            messages = generator.memory

        num_predict = base_model.num_predict
        base_model.num_predict = 1
        base_model.invoke(messages)
        base_model.num_predict = num_predict

    return seed


def ollama_check_and_pull_model(model_name: str) -> bool:
    """
    Ensure an Ollama model is available locally (pull if missing).

    :param model_name: Model name (may include 'ollama:' prefix).
    :type model_name: str
    :return: True if available or successfully pulled; False otherwise.
    :rtype: bool
    """
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[-1]
    try:
        # First, check if the model is available locally
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        # Check if the model name is in the output
        if model_name in result.stdout:
            return True

        # If not available locally, try to pull it
        logger.info(f"Model '{model_name}' not found locally. Pulling it from the hub...")
        pull_result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            check=True
        )

        if pull_result.returncode == 0:
            logger.info(f"Successfully pulled model '{model_name}'.")
            return True
        else:
            logger.error(f"Failed to pull model '{model_name}': {pull_result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error while pulling model '{model_name}' from ollama hub: {e}")
        return False


def make_serializable(data: dict) -> dict:
    """
    Convert non-JSON-serializable values in a dictionary to strings (in-place mutation).

    :param data: Dictionary to sanitize.
    :type data: dict
    :return: The mutated dictionary with serializable values.
    :rtype: dict
    :raises TypeError: If input is not a dict.
    """
    if type(data) is not dict:
        raise TypeError("Input must be a dictionary")

    for key, value in data.items():
        if hasattr(value, "json") and callable(value.json):
            data[key] = value.json()
        else:
            try:
                json.dumps(value)
            except (TypeError, OverflowError):
                data[key] = str(value)

    return data


def camel_or_snake_to_words(varname: str) -> str:
    """
    Convert camelCase or snake_case identifier into normalized spaced words.

    :param varname: Identifier string.
    :type varname: str
    :return: Human-readable spaced words.
    :rtype: str
    """
    # Replace underscores with spaces (snake_case)
    s = varname.replace('_', ' ')
    # Insert spaces before capital letters (camelCase, PascalCase)
    s = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', s)
    # Normalize multiple spaces
    return ' '.join(s.split())


def remove_audio_tags(text: str) -> str:
    """
    Remove tags of the form <...>. (Despite the summary mentioning {}, (), [], only angle brackets are removed.)

    :param text: Input text possibly containing markup tags.
    :type text: str
    :return: Text with angle-bracket tags removed.
    :rtype: str
    """
    return re.sub(r'<[^>]*>', '', text)


def dict_to_table(data: dict,
                  sort_by: str = None,
                  sort_ascending: bool = True,
                  markdown: bool = False,
                  format: str = ".2f",
                  show: bool = True) -> str:
    """
    Render a dict-of-dicts as a table (Markdown or fancy grid).

    :param data: Mapping where each value is itself a mapping of column -> value.
    :type data: dict
    :param sort_by: Column name to sort by.
    :type sort_by: str
    :param sort_ascending: Sort ascending if True.
    :type sort_ascending: bool
    :param markdown: If True produce GitHub-flavored Markdown table.
    :type markdown: bool
    :param format: Float formatting specifier passed to pandas.
    :type format: str
    :param show: If True print table to stdout.
    :type show: bool
    :return: Table as string.
    :rtype: str
    """
    if not data:
        return "(empty table)"
    df = pd.DataFrame(data).T
    df.index.name = "dataset"
    if sort_by:
        df.sort_values(by=sort_by, ascending=sort_ascending, inplace=True)
    if markdown:
        table = df.to_markdown(floatfmt=format)
    else:
        table = df.to_markdown(tablefmt='fancy_grid', floatfmt=format)

    if show:
        print(table)

    return table


def upper_camel_to_dash(name: str) -> str:
    """
    Convert UpperCamelCase to dash-case (preserving acronym groups).

    :param name: Class or identifier name.
    :type name: str
    :return: dash-case form.
    :rtype: str
    """
    # Improved to not split consecutive uppercase letters,
    # e.g., "HTTPServer" -> "http-server" instead of "h-t-t-p-server"
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1-\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', name)
    return name.lower()


class SentencePairTransformer:  # As opposed to SentenceTransformer
    """
    Transformer wrapper producing CLS embeddings for paired sentences (similar to NLI encoding).

    :param model_name: Hugging Face model name.
    :type model_name: str
    :param device: Explicit device (``"cpu"`` / ``"cuda:*"``); auto-detected if None.
    :type device: str
    :param verbose: Enable verbose progress display.
    :type verbose: bool
    """
    def __init__(self, model_name: str = "roberta-base", device: str = None, verbose: bool = True):
        """Initialize sentence pair encoder."""
        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device_name: {device}")
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.model.to(device)

    def encode(self,
               sent1: Union[str, List[str]],
               sent2: Union[str, List[str]],
               batch_size: int = 128,
               show_progress_bar: bool = True,
               progress_bar_desc: str = "Computing embeddings") -> np.ndarray:
        """
        Encode aligned sentence pairs into CLS embeddings.

        :param sent1: First sentence or list of first sentences.
        :type sent1: Union[str, List[str]]
        :param sent2: Second sentence or list of second sentences.
        :type sent2: Union[str, List[str]]
        :param batch_size: Batch size for encoding.
        :type batch_size: int
        :param show_progress_bar: Whether to show progress bar.
        :type show_progress_bar: bool
        :param progress_bar_desc: Description label for progress bar.
        :type progress_bar_desc: str
        :return: Array of shape (N, hidden_size) containing CLS embeddings.
        :rtype: np.ndarray
        """
        embs = []

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sent1, sent2, return_tensors='pt', padding=True, truncation=True)
            dataset = TensorDataset(*inputs.values())
            loader = DataLoader(dataset, batch_size=batch_size)
            for batch in tqdm(loader,
                              desc=progress_bar_desc,
                              disable=not show_progress_bar, leave=self.verbose):
                batch_inputs = batch_to_device({k: v for k, v in zip(inputs.keys(), batch)}, self.model.device)
                outputs = self.model(**batch_inputs)
                embs.append(outputs.last_hidden_state[:, 0].cpu().data)

        return torch.cat(embs).numpy()


class KNNModel:
    """
    Thin wrapper around sklearn NearestNeighbors for cosine similarity retrieval.

    :param items: Iterable of (item_id, embedding_vector) pairs.
    :type items: Iterable[Tuple[Any, Sequence[float]]]
    :param k: Default number of neighbors to retrieve.
    :type k: int
    """
    def __init__(self, items, k=3):
        """Initialize KNN index."""
        # items = (item, vector) pair list
        self.model = NearestNeighbors(algorithm='auto',
                                      metric="cosine",
                                      n_jobs=-1)
        self.k = k
        self.model.ix2id = {ix: item for ix, (item, _) in enumerate(items)}
        self.model.fit([vec for _, vec in items])

    def neighbors(self, target_emb, k=None):
        """
        Retrieve k nearest neighbors by cosine distance.

        :param target_emb: Query embedding vector.
        :type target_emb: Sequence[float]
        :param k: Override number of neighbors (defaults to self.k).
        :type k: int
        :return: List of (item_id, distance) pairs ordered by proximity.
        :rtype: List[Tuple[Any, float]]
        """
        k = k or self.k
        dists, indexes = self.model.kneighbors([target_emb],
                                               min(k, len(self.model.ix2id)),
                                               return_distance=True)
        dists, indexes = dists[0], indexes[0]
        return [(self.model.ix2id[indexes[ix]], dist) for ix, dist in enumerate(dists)]

    __call__ = neighbors


class CacheDialogScore:
    """
    Static class for caching utility for dialog scoring functions keyed by:
    (score class name, score object JSON-serializable attributes, dialog._path).

    Provides static methods to initialize, enable/disable, persist, and clear cache.
    """
    _cache = {}
    _score_obj_attributes = {}
    _cache_path = None
    _enable_cache = True

    @staticmethod
    def init(path, enable_cache=True):
        """
        Initialize cache system (load existing cache file if present).

        :param path: Directory path where cache file resides / will reside.
        :type path: str
        :param enable_cache: Whether to enable caching immediately.
        :type enable_cache: bool
        :return: None
        :rtype: None
        """
        cache_dir = os.path.expanduser(path)
        os.makedirs(cache_dir, exist_ok=True)
        CacheDialogScore.set_enable_cache(enable_cache)
        CacheDialogScore.set_cache_path(cache_dir)
        # Load cache dict if exists
        if os.path.exists(CacheDialogScore._cache_path):
            with open(CacheDialogScore._cache_path) as f:
                CacheDialogScore._cache = json.load(f)
        else:
            CacheDialogScore._cache = {}

    @staticmethod
    def set_enable_cache(enable: bool):
        """
        Enable or disable the cache.

        :param enable: True to enable caching, False to disable.
        :type enable: bool
        :return: None
        :rtype: None
        """
        CacheDialogScore._enable_cache = enable

    @staticmethod
    def is_cache_enabled() -> bool:
        """
        Check if caching is enabled.

        :return: True if enabled.
        :rtype: bool
        """
        return CacheDialogScore._enable_cache

    @staticmethod
    def get_cache():
        """
        Get internal cache dictionary.

        :return: Current in-memory cache mapping.
        :rtype: dict
        """
        return CacheDialogScore._cache

    @staticmethod
    def get_cache_path() -> str:
        """
        Get absolute path to cache JSON file.

        :return: Path to cache file.
        :rtype: str
        :raises ValueError: If init() not called first.
        """
        if CacheDialogScore._cache_path is None:
            raise ValueError("CacheDialogScore not initialized. Call CacheDialogScore.init(path) first.")
        return CacheDialogScore._cache_path

    @staticmethod
    def set_cache_path(path: str):
        """
        Set cache file path (creates directory if needed).

        :param path: Base directory for cache file.
        :type path: str
        :return: None
        :rtype: None
        """
        CacheDialogScore._cache_path = os.path.join(path, "dialog_scores_cache.json")
        if not os.path.exists(os.path.dirname(CacheDialogScore._cache_path)):
            os.makedirs(os.path.dirname(CacheDialogScore._cache_path), exist_ok=True)

    @staticmethod
    def save():
        """
        Persist cache dictionary to JSON file (if enabled).

        :return: None
        :rtype: None
        :raises ValueError: If not initialized.
        """
        if not CacheDialogScore.is_cache_enabled():
            logger.debug("CacheDialogScore is disabled, not saving cache.")
            return
        if CacheDialogScore._cache_path is None:
            raise ValueError("CacheDialogScore not initialized. Call CacheDialogScore.init(path) first.")
        os.makedirs(os.path.dirname(CacheDialogScore._cache_path), exist_ok=True)
        with open(CacheDialogScore._cache_path, "w") as f:
            json.dump(CacheDialogScore._cache, f)

    @staticmethod
    def cache(func):
        """
        Decorator adding disk-backed caching for dialog scoring functions.

        Cache key includes:

          * score object class name
          * JSON-serializable attributes of score object
          * dialog._path (must exist)

        :param func: Target scoring function ``(score_obj, dialog, *args, **kwargs)``.
        :type func: callable
        :return: Wrapped function with caching logic.
        :rtype: callable
        """
        @wraps(func)
        def wrapper(score_obj, dialog, *args, **kwargs):
            dialog_path = getattr(dialog, "_path", None)
            if not CacheDialogScore.is_cache_enabled() or dialog_path is None:
                result = func(score_obj, dialog, *args, **kwargs)
            else:
                score_obj_class = score_obj.__class__.__name__
                if score_obj_class not in CacheDialogScore._score_obj_attributes:
                    attrs = []
                    for attr in sorted(vars(score_obj)):
                        value = getattr(score_obj, attr)
                        try:
                            json.dumps(value)
                            attrs.append(attr)
                        except (TypeError, OverflowError):
                            continue
                    CacheDialogScore._score_obj_attributes[score_obj_class] = attrs
                else:
                    attrs = CacheDialogScore._score_obj_attributes[score_obj_class]
                attr_items = {attr: getattr(score_obj, attr) for attr in attrs}
                attr_str = json.dumps(attr_items, sort_keys=True)
                if (
                    score_obj_class in CacheDialogScore._cache
                    and attr_str in CacheDialogScore._cache[score_obj_class]
                    and dialog_path in CacheDialogScore._cache[score_obj_class][attr_str]
                ):
                    return CacheDialogScore._cache[score_obj_class][attr_str][dialog_path]
                result = func(score_obj, dialog, *args, **kwargs)
                if score_obj_class not in CacheDialogScore._cache:
                    CacheDialogScore._cache[score_obj_class] = {}
                if attr_str not in CacheDialogScore._cache[score_obj_class]:
                    CacheDialogScore._cache[score_obj_class][attr_str] = {}
                CacheDialogScore._cache[score_obj_class][attr_str][dialog_path] = result
            return result
        return wrapper

    @staticmethod
    def clear():
        """
        Clear in-memory cache and persist empty structure.

        :return: None
        :rtype: None
        """
        CacheDialogScore._cache = {}
        CacheDialogScore.save()
