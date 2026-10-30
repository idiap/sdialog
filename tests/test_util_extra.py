"""More tests for sdialog.util: model loading (with stubbed providers), Ollama helpers, seeding, caching, helpers."""
import json
import subprocess
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import sdialog.util as util
from sdialog import Dialog, Turn
from sdialog.util import (CacheDialogScore, KNNModel, dialogs_to_utt_pairs, dict_to_table, get_llm_default_params,
                          get_llm_model, make_serializable, ollama_check_and_pull_model, remove_audio_tags,
                          set_generator_seed, softmax, upper_camel_to_dash)


class FixedChatModel(BaseChatModel):
    seed: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


@pytest.fixture
def init_chat_model(monkeypatch):
    calls = []

    def fake_init_chat_model(name, **kwargs):
        calls.append((name, kwargs))
        return FixedChatModel()

    monkeypatch.setattr(util, "init_chat_model", fake_init_chat_model)
    return calls


@pytest.mark.parametrize("model_name, expected", [
    ("openai:gpt-4.1", "openai:gpt-4.1"),
    ("google:gemini-2.0-flash", "google_genai:gemini-2.0-flash"),
    ("anthropic:claude-sonnet-4-5", "anthropic:claude-sonnet-4-5"),
    ("azure:my-deployment", "azure_openai:my-deployment"),
])
def test_get_llm_model_providers(init_chat_model, model_name, expected):
    assert isinstance(get_llm_model(model_name, temperature=0.3), FixedChatModel)
    assert init_chat_model[-1] == (expected, {"temperature": 0.3})


def test_get_llm_model_bedrock_drops_seed(init_chat_model):
    get_llm_model("amazon:anthropic.claude-3", seed=1, temperature=0)
    assert init_chat_model[-1] == ("anthropic.claude-3", {"model_provider": "bedrock_converse", "temperature": 0})


def test_get_llm_model_ollama(monkeypatch):
    created = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(util, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(util, "ollama_check_and_pull_model", lambda name: True)
    monkeypatch.setattr(util, "get_llm_default_params", lambda name, params: {**params, "num_ctx": 2048})
    llm, params = get_llm_model("ollama:qwen3:8b", think=True, temperature=0.1, api_key="secret",
                                return_model_params=True)
    assert isinstance(llm, FakeChatOllama)
    assert created == {"model": "qwen3:8b", "reasoning": True, "temperature": 0.1, "api_key": "secret",
                       "num_ctx": 2048}
    assert params == {"temperature": 0.1, "num_ctx": 2048}  # secrets are not returned


def test_get_llm_model_instance_structured_output_and_tools():
    model = FixedChatModel()
    assert get_llm_model(model) is model
    with pytest.raises(ValueError):
        get_llm_model(object())

    class NoStructuredOutput:
        def invoke(self, messages):
            return "x"

    plain = NoStructuredOutput()
    assert get_llm_model(plain, output_format={"type": "object"}) is plain  # logged, returned as is

    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    class ToolModel(FixedChatModel):
        def bind_tools(self, tools, **kwargs):
            return ("bound", tools)

    assert get_llm_model(ToolModel(), tools=[add]) == ("bound", [add])


def _completed(stdout):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_get_llm_default_params(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("temperature 0.6\nnum_ctx 4096\nstop <x>\n"))
    params = get_llm_default_params("ollama:llama3", {"temperature": 0.2, "top_p": None})
    assert params == {"temperature": 0.2, "top_p": None, "num_ctx": 4096}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("num_ctx 4096\n"))
    assert get_llm_default_params("llama3", {})["temperature"] == 0.8
    assert get_llm_default_params("openai:gpt-4.1", {"a": 1}) == {"a": 1}

    def fail(*args, **kwargs):
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(subprocess, "run", fail)
    assert get_llm_default_params("llama3", {"temperature": 0.5}, retry=False) == {"temperature": 0.5}
    monkeypatch.setattr(subprocess, "Popen", fail)
    assert get_llm_default_params("llama3", {}) == {}  # retry fails to start the server too


def test_ollama_check_and_pull_model(monkeypatch):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return _completed("NAME\nllama3:latest\n" if cmd[1] == "list" else "pulled")

    monkeypatch.setattr(subprocess, "run", run)
    assert ollama_check_and_pull_model("ollama:llama3") is True and commands == [["ollama", "list"]]
    assert ollama_check_and_pull_model("qwen3") is True and commands[-1] == ["ollama", "pull", "qwen3"]
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: SimpleNamespace(stdout="", stderr="no", returncode=1))
    assert ollama_check_and_pull_model("qwen3") is False

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ollama")

    monkeypatch.setattr(subprocess, "run", fail)
    assert ollama_check_and_pull_model("qwen3") is False


def test_set_generator_seed():
    generator = SimpleNamespace(llm=FixedChatModel())
    assert set_generator_seed(generator, 7) == 7 and generator.llm.seed == 7
    bound = SimpleNamespace(llm=FixedChatModel().bind(stop=["\n"]))
    assert set_generator_seed(bound, 3) == 3 and bound.llm.bound.seed == 3
    chain = SimpleNamespace(llm=SimpleNamespace(steps=[FixedChatModel().bind(stop=["\n"])]))
    assert set_generator_seed(chain, 5) == 5
    assert set_generator_seed(SimpleNamespace(llm=object()), 5) is None  # not supported: torch is seeded instead
    assert isinstance(set_generator_seed(generator, None), int)


def test_set_generator_seed_ollama_warmup(monkeypatch):
    calls = []

    class FakeOllama(util.ChatOllama):
        def invoke(self, messages, *args, **kwargs):
            calls.append((self.num_predict, messages))

    llm = FakeOllama(model="llama3", num_predict=64)
    set_generator_seed(SimpleNamespace(llm=llm, messages=["m"]), 1)
    set_generator_seed(SimpleNamespace(llm=llm, memory=["memory"]), 2)
    assert calls == [(1, ["m"]), (1, ["memory"])] and llm.num_predict == 64


def test_dialogs_to_utt_pairs():
    dialog = Dialog(turns=[Turn(speaker="User", text="a"), Turn(speaker="AI", text="b"),
                           Turn(speaker="AI", text="c"), Turn(speaker="User", text="d")])
    assert dialogs_to_utt_pairs([dialog]) == (["a", "b", "c"], ["b", "c", "d"])
    assert dialogs_to_utt_pairs([dialog, Dialog(turns=[Turn(speaker="User", text="x")])], ai_speaker="ai") == (
        ["a"], ["b"])
    with pytest.raises(ValueError, match="AI speaker"):
        dialogs_to_utt_pairs([dialog], ai_speaker="nobody")
    with pytest.raises(ValueError, match="valid turns"):
        dialogs_to_utt_pairs([Dialog(turns=[])])


def test_small_helpers(capsys):
    class WithJson:
        def json(self):
            return {"a": 1}

    data = make_serializable({"obj": WithJson(), "set": {1}, "ok": [1, "x"]})
    assert data == {"obj": {"a": 1}, "set": "{1}", "ok": [1, "x"]}
    with pytest.raises(TypeError):
        make_serializable([1])
    assert remove_audio_tags("Hi <laugh>there</laugh>") == "Hi there"
    assert upper_camel_to_dash("HTTPServerError") == "http-server-error"
    assert softmax([1, 1]) == pytest.approx([0.5, 0.5])
    assert dict_to_table({}) == "(empty table)"
    table = dict_to_table({"a": {"x": 2.0}, "b": {"x": 1.0}}, sort_by="x", markdown=True)
    assert table.index("| b") < table.index("| a") and "| b" in capsys.readouterr().out
    assert "╒" in dict_to_table({"a": {"x": 1}}, show=False)


def test_knn_model():
    knn = KNNModel([("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [0.7, 0.7])], k=2)
    assert [n for n, _ in knn([1.0, 0.1])] == ["a", "c"]
    batch = knn(np.array([[1.0, 0.0], [0.0, 1.0]]), k=1)
    assert [[n for n, _ in neighbors] for neighbors in batch] == [["a"], ["b"]]
    assert len(knn.neighbors([[0.0, 1.0]], k=5)[0]) == 3  # k is capped by the number of items


@pytest.fixture
def cache(tmp_path):
    CacheDialogScore.init(str(tmp_path / "cache"), enable_cache=True)
    yield tmp_path / "cache" / "dialog_scores_cache.json"
    CacheDialogScore.set_enable_cache(False)
    CacheDialogScore._cache = {}
    CacheDialogScore._cache_path = None
    CacheDialogScore._score_obj_attributes = {}


def test_cache_dialog_score(cache):
    calls = []

    class CountingScore:
        def __init__(self):
            self.threshold = 0.5  # part of the cache key
            self.not_json = object()  # not part of the cache key

        @CacheDialogScore.cache
        def score(self, dialog):
            calls.append(dialog)
            return len(dialog)

    dialog = Dialog(turns=[Turn(speaker="A", text="hi")])
    dialog._path = "/data/dialog.json"
    score = CountingScore()
    assert score.score(dialog) == 1 and score.score(dialog) == 1 and len(calls) == 1
    score.score(Dialog(turns=[]))  # no path: not cached
    assert len(calls) == 2
    assert CacheDialogScore.is_cache_enabled() and CacheDialogScore.get_cache()
    CacheDialogScore.save()
    assert json.load(open(cache))["CountingScore"]
    CacheDialogScore.init(str(cache.parent), enable_cache=True)  # loads the saved cache
    assert CacheDialogScore.get_cache_path() == str(cache)
    assert "/data/dialog.json" in json.dumps(CacheDialogScore.get_cache())
    CacheDialogScore.clear()
    assert json.load(open(cache)) == {}


def test_cache_dialog_score_not_initialized():
    CacheDialogScore._cache_path = None
    with pytest.raises(ValueError):
        CacheDialogScore.get_cache_path()
    CacheDialogScore.set_enable_cache(True)
    try:
        with pytest.raises(ValueError):
            CacheDialogScore.save()
    finally:
        CacheDialogScore.set_enable_cache(False)
    CacheDialogScore.save()  # disabled: nothing to do


def test_dynamic_version_without_git(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", fail)
    assert util._get_dynamic_version() == util.__version__
