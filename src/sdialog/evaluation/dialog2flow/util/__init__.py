# -*- coding: utf-8 -*-
"""
Utility functions and SentenceTransformer wrappers for baseline models.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import re
import numpy as np
import pandas as pd

from ollama import chat
from joblib import Memory
from openai import OpenAI
from tqdm.autonotebook import trange
from sentence_transformers import util
from tenacity import retry, wait_random_exponential, stop_after_attempt


STR_MODEL_COLUMN = "Model"
STR_AVERAGE_COLUMN = "AVG."

memory = Memory('__cache__/llm_labels', verbose=0)
gpt_client = None
cluster_label_prompt = """Your task is to annotate conversational utterances with the intent expressed as canonical forms. A canonical form is a short summary representing the intent of a set of utterances - it is neither too verbose nor too short.
Be aware that required canonical forms should avoid containing specific names or quantities, only represent the intent in abstract terms.
For example, for:

For the following utterances:
    1. Uh yes i'm looking for a place for entertainment that is in the center of the city
    2. i would like to know where a place for entertainment that is not far away from my location
Canonical form is: "request entertainment place and inform location"

For the following utterances:
    1. My name is Michael.
    2. I'm John
    3. mmmh sure, it's Dr. Smith
    4. Angela
Canonical form is: "inform name"

For the following utterances:
    1. Okay so the phone number is a 1223217297
    2. Sure, my phone number is four four five five
    3. 2 3 4 5 6 is her phone number
Canonical form is: "inform phone number"

For the following utterances:
    1. 8 4 0
    2. yes five five three
Canonical form is: "inform number"

For the following utterances:
    1. Is there anything else that I can do for you?
    2. anything else I can help you with?
Canonical form is: "ask anything else"

For the following utterances:
    1. Thank you and goodbye
    2. Yes. Thank you for your help. Have a nice day
    3. Thanks goodbye
Canonical form is: "good bye"

For the following utterances:
    1. I'm just trying to check up on the status of payment. Um, I had requested to reopen this claim. It was approved. Um, I, it was assigned an adjuster and then reassigned an adjuster and then I sent emails to the adjusters, their supervisors and the directors and I still have not been able to get any kind of, uh, status update.
    2. Uh I, I don't understand how it would be closed. We did an inspection, um and we never got any response, any calls, any, anything. So, I mean, I don't understand how it would be closed. They never sent us a field adjuster report, a denial letter. Nothing. That's, that's what I'm saying. I, I this claim was filed in like April and I've never heard anything from Jane whatsoever. I just thought after multiple, multiple emails, I finally got a call from the field adjuster and that was two months ago, we completed that inspection a month ago. And then, you know, it's been crickets ever since I sent her multiple follow ups and there's been absolutely nothing. So yes, Absolutely, I definitely need supplemental. I mean, a coverage decision at this point. To be honest,
    3. Ok. Um here's the problem the adjuster she got that done her last day.
    4. Um I haven't I just haven't um this every time I call or email Miss June um she says she's missing some information from you guys.
Canonical form is: "problem statement"
"""  # noqa: E501


class CaselessDict(dict):
    def __setitem__(self, key, value):
        super(CaselessDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaselessDict, self).__getitem__(key.lower())


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_response(client, messages: list, model="gpt-4o", seed=42):
    if isinstance(gpt_client, OpenAI):
        response = client.chat.completions.create(
            seed=seed,
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    else:
        response = chat(
            model=model,
            messages=messages,
            options={"seed": seed, "stop": ['"']})
        return response['message']['content']


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_embedding(client, docs: list, model="text-embedding-3-large", dimensions=768) -> list[float]:
    if isinstance(docs, np.ndarray):
        docs = docs.tolist()
    data = client.embeddings.create(input=docs, model=model, dimensions=dimensions).data
    if len(data) > 1:
        return [d.embedding for d in data]
    return data[0].embedding


def init_gpt(model_name="gpt-4-turbo-2024-04-09", seed=42):
    global gpt_client, gpt_model, gpt_seed
    if gpt_client is None and "gpt" in model_name:
        gpt_client = OpenAI()
    gpt_model = model_name
    gpt_seed = seed


@memory.cache
def get_cluster_label(utterances, llm_model_name):
    messages = [
        {"role": "system", "content": cluster_label_prompt},
        {"role": "user", "content": ("Give the following list of utterance provide a single canonical "
                                     "name that represent all of them:\n{utts}").replace(
                                         "{utts}",
                                         "\n".join(f"{ix + 1}. {utt}" for ix, utt in enumerate(utterances)))},
        {"role": "assistant", "content": 'Canonical form is: "'}
    ]
    response = get_openai_response(gpt_client,
                                   messages,
                                   model=gpt_model,
                                   seed=gpt_seed)
    m = re.match(r'.+?:\s*"(.+?)".*', response)
    if m:
        response = m.group(1)
    else:
        m = re.match(r'.+?:\s*"(.+)', response)
        if m:
            response = m.group(1)
    return response.strip('"').capitalize()


def slugify(text):
    if "outputs/" in text:
        text = text.split("outputs/")[1]
    return "-".join(re.findall(r"\w+", text))


def get_turn_text(turn: dict, use_ground_truth: bool = False):
    if use_ground_truth:
        # (id, speaker, acts)
        if not turn["turn"] or ":" not in turn["turn"]:
            return "unknown"
        dial_act = turn["turn"].split(": ")[1]
        if re.match(r"^\w+-(\w)", dial_act):
            return dial_act.split("-")[1]
        return dial_act
    return turn["text"]


# https://aclanthology.org/D19-1006.pdf
# https://aclanthology.org/2022.emnlp-main.603.pdf
def compute_anisotropy(embs):
    sim = util.cos_sim(embs, embs)
    sim.fill_diagonal_(0)
    return (sim.sum() / (sim.shape[0] ** 2 - sim.shape[0])).item()


def get_print_column_value(row, column, percentage=False, extra_value=None):
    value = row[column]
    if percentage:
        value = f"{value:.2%}"
    else:
        value = f"{value:.3f}"
    extra_value = f"+{extra_value}" if extra_value and extra_value > 0 else extra_value
    return f"{value} ({extra_value})" if extra_value is not None else value


def show_results(models, domains, score_getter, metric_name="",
                 metric_is_ascending=False, print_table=True, sorted=False,
                 percentage=False, value_extra_getter=None, column_value_getter=None):
    rows = []
    columns = [f"{dom}_{metric_name}" for dom in domains]
    for model in models:
        row = {STR_MODEL_COLUMN: model}
        for ix, column in enumerate(columns):
            domain = domains[ix]
            if domain != STR_AVERAGE_COLUMN:
                row[column] = score_getter(model, domain)
            else:
                row[column] = sum(row[col] for col in columns if STR_AVERAGE_COLUMN not in col) / (len(domains) - 1)
        rows.append(row)

    df = pd.DataFrame.from_dict(rows)
    columns = df.columns[1:]
    for column in columns:
        ranking = df[column].sort_values(ascending=metric_is_ascending).tolist()
        ranking = list(dict.fromkeys(ranking).keys())  # Removing duplicates
        df[f"{column}_rank"] = df[column].map(lambda v: ranking.index(v) + 1)

    avg_ranking_column = f"{STR_AVERAGE_COLUMN}_{metric_name}"

    if sorted:
        df.sort_values(by=[avg_ranking_column], ascending=metric_is_ascending, inplace=True)

    if print_table:
        print_table = []
        for _, row in df.iterrows():
            print_row = {STR_MODEL_COLUMN.upper(): row[STR_MODEL_COLUMN]}
            for dom in domains:
                column_extra_value = column_value_getter(row[STR_MODEL_COLUMN], dom)
                col_name = f"{dom} ({column_extra_value})" if column_extra_value else dom
                print_row[col_name] = get_print_column_value(
                    row,
                    column=f"{dom}_{metric_name}",
                    percentage=percentage,
                    extra_value=value_extra_getter(row[STR_MODEL_COLUMN], dom) if value_extra_getter else None
                )
            print_table.append(print_row)
        print(pd.DataFrame.from_dict(print_table).to_markdown(index=False))

    return df


class SentenceTransformerOpenAI():
    """Simple SentenceTransformer wrapper for OpenAI embedding models"""
    def __init__(self, model_name):
        self.client = OpenAI()
        self.model = model_name

    def __call__(self, features):
        return self.forward(features)

    def forward(self, features):
        embedding = get_openai_embedding(self.client, features, model=self.model)
        return {'sentence_embedding': embedding}

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index:start_index + batch_size]
            embeddings = self.forward(sentences_batch)[output_value]
            all_embeddings.extend(embeddings)

        if convert_to_numpy:
            all_embeddings = np.asarray([np.array(emb) for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
