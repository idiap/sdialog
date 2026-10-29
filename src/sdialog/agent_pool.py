"""
Parallel serving of an agent, running each conversation turn on a fresh clone (used by the SDialog servers).
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import asyncio
import logging
import functools

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .agents import Agent

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 8


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
    Runs conversation turns of an agent in parallel, up to ``max_workers`` at a time, each on a fresh clone of the
    agent (see :meth:`~sdialog.agents.Agent.clone`), which shares its model and is discarded after the turn.

    Turns with the same key (e.g. a user ID) run one at a time and in arrival order, so a user's messages are never
    processed concurrently, while turns with different keys run in parallel. Callers restore the conversation on
    the lent agent before each turn (e.g. with :meth:`~sdialog.agents.Agent.set_state`) and save it afterwards.

    Parallel turns only help when the model runs elsewhere (an API or a model server), so agents with a local
    Hugging Face model or with inspectors (whose hooks would mix activations from different conversations) are
    served one turn at a time. With a single worker, the agent itself is lent instead of a clone.

    Agent calls run in the pool's own threads (see :meth:`run`). With several workers, the agent's tools and
    custom functions (e.g. ``postprocess_fn``) may run at the same time for different conversations, so they must
    be thread-safe; use ``max_workers=1`` otherwise.

    Example:

        .. code-block:: python

            from sdialog.agents import Agent
            from sdialog.agent_pool import AgentPool

            pool = AgentPool(Agent(name="Support"))

            async def turn(user_id, state, message):
                async with pool.acquire(user_id) as agent:
                    agent.reset()
                    if state:
                        agent.set_state(state)
                    reply = await pool.run(agent, message)
                    return reply, agent.get_state()

    :param agent: The agent to serve.
    :type agent: Agent
    :param max_workers: Maximum number of turns processed at the same time. If None, 8, or 1 for agents with a
                        local model or inspectors.
    :type max_workers: Optional[int]
    """

    def __init__(self, agent: Agent, max_workers: Optional[int] = None):
        if max_workers is not None and \
           (isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1):
            raise ValueError(f"max_workers must be a positive integer or None, got {max_workers!r}")

        sequential_reason = None
        if uses_local_model(agent):
            sequential_reason = "runs a local model"
        elif getattr(agent, "_inspectors", None):
            sequential_reason = "has inspectors"
        if sequential_reason:
            if max_workers is not None and max_workers > 1:
                logger.warning(f"Agent '{agent.get_name()}' {sequential_reason}: serving one turn at a time "
                               f"(max_workers={max_workers} ignored)")
            max_workers = 1

        self.agent = agent
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="sdialog-agent")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._key_locks: Dict[str, List] = {}  # key -> [lock, number of turns holding or waiting for it]

    def _bind_loop(self) -> None:
        # asyncio primitives are bound to an event loop, so they are created lazily in the one running the server
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_workers)
            self._key_locks = {}

    @asynccontextmanager
    async def acquire(self, key: Optional[str] = None) -> AsyncIterator[Agent]:
        """
        Wait for a free worker and lend an agent for one turn: a fresh clone, or the agent itself with a single
        worker.

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
            yield self.agent if self.max_workers == 1 else self.agent.clone()

    async def run(self, func: Callable, *args) -> Any:
        """
        Run a blocking call (e.g. ``agent(message)``) in one of the pool's threads, without blocking the event loop.

        The pool has its own ``max_workers`` threads, so the number of turns running at the same time is not
        limited by asyncio's default thread pool.

        :param func: The function to call.
        :type func: Callable
        :return: The function's result.
        """
        return await asyncio.get_running_loop().run_in_executor(self._executor, functools.partial(func, *args))

    def close(self) -> None:
        """Release the pool's threads once running calls finish."""
        self._executor.shutdown(wait=False)
