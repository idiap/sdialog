"""
Tests for Agent.clone() and AgentPool (parallel serving), including the REST API server.
"""
import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sdialog.agents import Agent, _copy_llm
from sdialog.agent_pool import DEFAULT_MAX_WORKERS, AgentPool, uses_local_model
from sdialog.orchestrators import ChangeMindOrchestrator
from sdialog.personas import Persona


class EchoChatModel(BaseChatModel):
    """Chat model replying with the user messages it got, after ``latency`` seconds."""

    seed: Optional[int] = None
    latency: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.latency)
        said = [m.content for m in messages if isinstance(m, HumanMessage)]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="echo: " + " | ".join(said)))])


def _agent(latency=0.0, **kwargs):
    return Agent(persona=Persona(name="Bot"), name="Bot", model=EchoChatModel(latency=latency), **kwargs)


# ---------------------------------------------------------------------------
# Agent.clone()
# ---------------------------------------------------------------------------

def test_clone_has_its_own_conversation():
    agent = _agent()
    clone = agent.clone()

    assert agent("I'm Alice") == "echo: I'm Alice"
    assert clone("I'm Bob") == "echo: I'm Bob"  # Does not see Alice
    assert agent("Who am I?") == "echo: I'm Alice | Who am I?"
    assert clone.prompt() == agent.prompt() and clone.memory[0] is not agent.memory[0]
    assert clone.persona is agent.persona


def test_clone_shares_model_but_not_llm_settings():
    agent = _agent()
    clone = agent.clone()

    assert type(clone.llm) is type(agent.llm) and clone.llm is not agent.llm
    clone.llm.seed = 123
    assert agent.llm.seed != 123

    binding = EchoChatModel().bind(stop=["END"])  # e.g. a model with bound tools
    copied = _copy_llm(binding)
    assert copied.bound is not binding.bound and copied.kwargs == binding.kwargs


def test_clone_reset_does_not_change_original_prompt():
    agent = _agent()
    prompt = agent.prompt()
    clone = agent.clone()
    clone.reset(context="A completely different place")
    assert agent.prompt() == prompt
    assert clone.prompt() != prompt


def test_clone_copies_orchestrators():
    agent = _agent() | ChangeMindOrchestrator(max_times=2)
    agent._orchestrators[0].times = 1
    clone = agent.clone()

    orchestrator = clone._orchestrators[0]
    assert orchestrator is not agent._orchestrators[0]
    assert orchestrator._target is clone
    orchestrator.times = 2
    assert agent._orchestrators[0].times == 1
    assert orchestrator.reasons is agent._orchestrators[0].reasons  # Shallow copy: resources are shared


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------

def _run_turns(pool, turns):
    """Run (key, message) turns concurrently; return (replies in completion order, elapsed seconds)."""
    replies = []

    async def turn(key, message):
        async with pool.acquire(key) as agent:
            agent.reset()
            replies.append(await pool.run(agent, message))

    async def main():
        await asyncio.gather(*(turn(key, message) for key, message in turns))

    start = time.monotonic()
    asyncio.run(main())
    return replies, time.monotonic() - start


def test_pool_runs_different_keys_in_parallel():
    pool = AgentPool(_agent(latency=0.5), max_workers=3)
    replies, elapsed = _run_turns(pool, [("a", "1"), ("b", "2"), ("c", "3")])
    assert sorted(replies) == ["echo: 1", "echo: 2", "echo: 3"]
    assert elapsed < 1.0  # One at a time would take 1.5s


def test_pool_runs_same_key_in_order():
    pool = AgentPool(_agent(latency=0.2), max_workers=3)
    replies, elapsed = _run_turns(pool, [("a", "1"), ("a", "2"), ("a", "3")])
    assert replies == ["echo: 1", "echo: 2", "echo: 3"]
    assert elapsed >= 0.6
    assert pool._key_locks == {}  # Locks of finished keys are dropped


def test_pool_limits_parallel_turns():
    pool = AgentPool(_agent(latency=0.4), max_workers=2)
    _, elapsed = _run_turns(pool, [("a", "1"), ("b", "2"), ("c", "3")])
    assert elapsed >= 0.8  # The third turn waits for a free worker


def _lent_agents(pool, n):
    async def main():
        lent = []
        for _ in range(n):
            async with pool.acquire() as agent:
                lent.append(agent)
        return lent
    return asyncio.run(main())


def test_pool_lends_fresh_clones():
    agent = _agent()
    pool = AgentPool(agent, max_workers=2)
    lent = _lent_agents(pool, 3)
    assert len({id(a) for a in lent}) == 3 and agent not in lent

    _run_turns(pool, [("a", "1"), ("b", "2")])
    assert len(agent.memory) == 1  # The served agent itself is never used


def test_pool_single_worker_lends_the_agent_itself():
    agent = _agent()
    assert _lent_agents(AgentPool(agent, max_workers=1), 2) == [agent, agent]


def test_pool_default_max_workers():
    assert AgentPool(_agent()).max_workers == DEFAULT_MAX_WORKERS == 8


def test_pool_threads_are_not_limited_by_asyncio_default_executor():
    pool = AgentPool(_agent(latency=0.5), max_workers=40)
    _, elapsed = _run_turns(pool, [(str(i), str(i)) for i in range(40)])
    assert elapsed < 1.0  # asyncio's default executor has at most 32 threads (fewer on small machines)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_pool_rejects_invalid_max_workers(value):
    with pytest.raises(ValueError, match="max_workers"):
        AgentPool(_agent(), max_workers=value)


def test_pool_uses_one_worker_for_local_models_and_inspectors(caplog):
    local = SimpleNamespace(llm=SimpleNamespace(pipeline=object()), get_name=lambda: "Local")
    assert uses_local_model(local)
    assert AgentPool(local).max_workers == 1
    assert caplog.text == ""  # No warning when max_workers was not asked for
    assert AgentPool(local, max_workers=4).max_workers == 1
    assert "runs a local model" in caplog.text

    inspected = _agent()
    inspected._inspectors = [object()]
    assert AgentPool(inspected, max_workers=4).max_workers == 1
    assert "has inspectors" in caplog.text

    assert not uses_local_model(_agent())


# ---------------------------------------------------------------------------
# REST API server
# ---------------------------------------------------------------------------

@pytest.fixture
def rest_server():
    from sdialog.server import Server

    saved = {name: getattr(Server, name) for name in ("_stateless", "_debug", "_rate_limiter", "_max_workers")}
    Server._agents.clear()
    Server._agent_locks.clear()
    Server._pools = {}
    Server._app = None
    Server._rate_limiter = None
    yield Server
    Server._agents.clear()
    Server._agent_locks.clear()
    Server._pools = {}
    Server._app = None
    for name, value in saved.items():
        setattr(Server, name, value)


def _post_concurrently(client, messages):
    responses = {}

    def post(message):
        responses[message] = client.post("/v1/chat/completions", json={
            "model": "bot:latest", "messages": [{"role": "user", "content": message}]})

    threads = [threading.Thread(target=post, args=(message,)) for message in messages]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return responses, time.monotonic() - start


def test_rest_server_serves_requests_in_parallel(rest_server):
    from fastapi.testclient import TestClient

    rest_server._setup_agents(_agent(latency=0.5), "bot", stateless=True)
    rest_server._set_max_workers(2)
    with TestClient(rest_server._app) as client:
        responses, elapsed = _post_concurrently(client, ["Hi from A", "Hi from B"])

    for message, response in responses.items():
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == f"echo: {message}"
    assert elapsed < 0.9  # One at a time would take 1s


def test_rest_server_keeps_answering_during_generation(rest_server):
    from fastapi.testclient import TestClient

    rest_server._setup_agents(_agent(latency=0.6), "bot", stateless=True)
    rest_server._set_max_workers(1)
    with TestClient(rest_server._app) as client:
        slow = threading.Thread(target=_post_concurrently, args=(client, ["Hi"]))
        slow.start()
        time.sleep(0.2)
        start = time.monotonic()
        assert client.get("/health").status_code == 200
        assert time.monotonic() - start < 0.3  # Not blocked by the running generation
        slow.join()


def test_rest_server_stateful_agents_use_one_worker(rest_server):
    rest_server._setup_agents(_agent(), "bot", stateless=False)
    rest_server._set_max_workers(4)
    assert rest_server._get_pool("bot:latest").max_workers == 1

    rest_server._stateless = True
    rest_server._set_max_workers(4)
    assert rest_server._get_pool("bot:latest").max_workers == 4
    rest_server._set_max_workers(None)
    assert rest_server._get_pool("bot:latest").max_workers == DEFAULT_MAX_WORKERS


def test_rest_server_rebuilds_pool_for_replaced_agent(rest_server):
    first, second = _agent(), _agent()
    rest_server._setup_agents(first, "bot", stateless=True)
    assert rest_server._get_pool("bot:latest").agent is first
    rest_server.add_agent(second, "bot")
    assert rest_server._get_pool("bot:latest").agent is second
