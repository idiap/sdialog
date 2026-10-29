"""
Pool of agent clones to serve several conversations in parallel (used by the SDialog servers).
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import asyncio
import logging

from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional

from .agents import Agent

logger = logging.getLogger(__name__)


def uses_local_model(agent: Agent) -> bool:
    """
    Whether the agent runs a local (Hugging Face) model in this process, rather than calling a model server or API.

    :param agent: The agent.
    :type agent: Agent
    :rtype: bool
    """
    llm = getattr(agent, "llm", None)
    llm = getattr(llm, "bound", llm)  # Unwrap bound tools
    return hasattr(llm, "pipeline") or hasattr(getattr(llm, "llm", None), "pipeline")


class AgentPool:
    """
    Pool of agents (the given agent plus clones of it sharing its model) to run up to ``max_workers``
    conversation turns at a time.

    Turns with the same key (e.g. a user ID) run one at a time and in arrival order, so a user's messages are
    never processed concurrently, while turns with different keys run in parallel on different agents.
    Agents are lent as they are, so callers restore the state of the conversation before each turn
    (e.g. with :meth:`~sdialog.agents.Agent.set_state`).

    Parallel turns only help when the model runs elsewhere (an API or model server), so the pool is limited to
    a single agent, with a warning, for local Hugging Face models or agents with inspectors (whose hooks
    would mix activations from different conversations).

    Example:

        .. code-block:: python

            import asyncio
            from sdialog.agents import Agent
            from sdialog.agent_pool import AgentPool

            pool = AgentPool(Agent(name="Support"), max_workers=4)

            async def turn(user_id, state, message):
                async with pool.acquire(user_id) as agent:
                    agent.reset()
                    if state:
                        agent.set_state(state)
                    reply = await asyncio.to_thread(agent, message)
                    return reply, agent.get_state()

    :param agent: The agent to serve (used as the first worker).
    :type agent: Agent
    :param max_workers: Maximum number of turns processed at the same time.
    :type max_workers: int
    """

    def __init__(self, agent: Agent, max_workers: int = 1):
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers!r}")
        if max_workers > 1:
            if uses_local_model(agent):
                logger.warning(f"Agent '{agent.get_name()}' runs a local model: serving one turn at a time "
                               f"(max_workers={max_workers} ignored)")
                max_workers = 1
            elif getattr(agent, "_inspectors", None):
                logger.warning(f"Agent '{agent.get_name()}' has inspectors: serving one turn at a time "
                               f"(max_workers={max_workers} ignored)")
                max_workers = 1

        self.agent = agent
        self.max_workers = max_workers
        self._workers: List[Agent] = [agent] + [agent.clone() for _ in range(max_workers - 1)]
        self._free: List[Agent] = list(self._workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._key_locks: Dict[str, List] = {}  # key -> [lock, number of turns holding or waiting for it]

    def _bind_loop(self) -> None:
        # asyncio primitives are bound to an event loop, so they are created lazily in the one running the server
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(len(self._free))
            self._key_locks = {}

    @asynccontextmanager
    async def acquire(self, key: Optional[str] = None) -> AsyncIterator[Agent]:
        """
        Wait for a free agent and lend it for one turn.

        :param key: Conversation key (e.g. user ID). Turns with the same key run one at a time, in order.
                    If None, the turn is not ordered with respect to any other.
        :type key: Optional[str]
        :return: Async context manager yielding the agent.
        """
        self._bind_loop()
        if key is None:
            async with self._lend() as agent:
                yield agent
            return

        entry = self._key_locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:  # Wait for this key's previous turns before taking a worker
                async with self._lend() as agent:
                    yield agent
        finally:
            entry[1] -= 1
            if entry[1] == 0 and self._key_locks.get(key) is entry:
                del self._key_locks[key]

    @asynccontextmanager
    async def _lend(self) -> AsyncIterator[Agent]:
        async with self._semaphore:
            agent = self._free.pop()
            try:
                yield agent
            finally:
                self._free.append(agent)
