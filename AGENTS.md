# AGENTS.md

Guidance for coding agents working on **SDialog**, a Python toolkit for building, simulating, evaluating and serving LLM-based conversational agents. Human-oriented docs: [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), https://sdialog.readthedocs.io (an LLM-friendly summary is at https://sdialog.readthedocs.io/en/latest/llm.txt).

If anything here conflicts with what the user asks, follow the user.

## Repository layout

```
src/sdialog/            The package (published to PyPI as "sdialog")
  __init__.py           Core data model: Dialog, Turn, Event, Context, Instruction
  agents.py             Agent: persona-driven LLM agent (memory, tools, orchestrators, inspectors, state, clone)
  personas.py           Persona classes (BasePersona = base.BaseAttributeModel)
  base.py               BaseAttributeModel, Metadata
  orchestrators/        BaseOrchestrator, BasePersistentOrchestrator and built-in orchestrators
  generators/           Persona/context/dialog generators, Paraphraser
  evaluation/           Metrics, LLM judges, dataset evaluators (and dialog2flow)
  interpretability/     Inspector, hooks and steerers (local Hugging Face models only)
  server.py             OpenAI/Ollama-compatible REST server (Server, Agent.serve)
  whatsapp/             WhatsApp Cloud API server (Agent.serve_whatsapp)
  agent_pool.py         AgentPool: parallel serving with agent clones
  rate_limit.py         RateLimiter shared by both servers
  audio/                Text-to-speech and acoustic simulation pipeline (optional extra)
  config/               Global config (config.yaml) and helpers: sdialog.config.llm(...), set_cache(...)
  prompts/              Jinja2 prompt templates
  util.py               LLM loading (get_llm_model), helpers, and __version__
tests/                  pytest suite (conftest.py stubs optional audio deps)
docs/                   Sphinx docs (ReadTheDocs); API reference modules listed by hand in docs/api/sdialog.rst
tutorials/              Notebooks (keep outputs cleared)
```

## Setup

Python >= 3.9 (CI runs 3.9 and 3.10 on Ubuntu 24.04). Dependencies are read from `requirements.txt` by `pyproject.toml`.

```bash
pip install -e .                 # core
pip install -e .[audio]          # audio extra (requirements-audio.txt; needs system deps, see docs)
pip install -r requirements-audio-test.txt   # what CI installs to run the audio tests
```

Don't upgrade core ML libraries (`torch`, `transformers`, `langchain*`) as a side effect of another change, and don't install CUDA wheels.

## Commands (match CI in .github/workflows/ci.yml)

```bash
flake8 --ignore=W503 --max-line-length=120 --exclude=./tutorials/,./src/sdialog/audio/tts/dia2/_dia2/
pytest -v                                       # whole suite
pytest -v --cov=src/sdialog --cov-report=term-missing
cd docs && python -m sphinx -b html . ../docs_html   # docs (pip install -r docs/requirements.txt)
```

The test suite must not call external APIs or need a GPU. Tests that do are skipped unless `SDIALOG_RUN_LIVE=1` (see `tests/test_audio_dia2_live.py`).

## Conventions

- **Style:** flake8 with 120-character lines. Match the surrounding code's naming, comment density and idioms; keep diffs minimal and don't reformat unrelated code.
- **New files** start with the SPDX header used across `src/` (`SPDX-FileCopyrightText`, `SPDX-FileContributor`, `SPDX-License-Identifier: MIT`). When adapting a contributor's code, keep their name as a contributor.
- **Public API:** reST docstrings (`:param:`, `:type:`, `:return:`) with an `Example:` block for new public classes and methods; type hints on new public signatures; keep backward compatibility (add parameters at the end, with defaults that preserve current behavior).
- **Docs:** user-visible changes go in `docs/sdialog/index.rst` (narrative) and, for new modules, `docs/api/sdialog.rst`.
- **Changelog:** add user-facing changes to `CHANGELOG.md` under `[Unreleased]`, grouped as Added / Changed / Fixed / Deprecated / Removed / Security. Behavior changes for existing users go under Changed.
- **Tests:** add tests for new public behavior and bug fixes (a regression test that fails without the fix). Use fake LLMs (a small LangChain `BaseChatModel` subclass, or the `DummyLLM` pattern in `tests/test_agents.py`) instead of real models; for randomness, fix seeds and assert structure, not exact text.
- **Imports:** keep module import cheap; guard optional backends (audio, cloud SDKs) and import them where used.
- **Commits:** a short imperative summary line (no prefix), plus an optional body explaining why. One logical change per commit.

## LLM configuration

```python
import sdialog
sdialog.config.llm("openai:gpt-4.1", temperature=0.7)   # global default model
sdialog.config.llm_params(max_tokens=512)
```

Model names use a provider prefix: `openai:`, `azure:` (Azure OpenAI), `anthropic:`, `google:` (Gemini), `amazon:` or `aws:` (Bedrock), `ollama:`, `huggingface:` (loaded locally). Without a known prefix, names containing `/` are treated as Hugging Face and the rest as Ollama. A LangChain model instance can also be passed as `model=`. Classes that create models accept an explicit `model=` and fall back to `config["llm"]["model"]`. API keys come from environment variables; never hard-code, log or commit secrets.

## Things that are easy to get wrong

- **`agent | component` attaches** an orchestrator or inspector to the agent itself and returns it; it does not copy. Use `agent.clone()` for an independent copy that shares the model.
- **Agent state:** `get_state()` / `set_state()` save and restore a conversation (memory without the system prompt, finished flag, orchestrator state). Stateful orchestrators implement their own `get_state()` / `set_state()`.
- **`Agent.reset()` reseeds the LLM** (and, for Ollama, runs a one-token generation with `num_predict` temporarily set to 1), mutating the LLM object. That's why clones get their own copy of the LangChain wrapper.
- **Interpretability** (`Inspector`, steering) only works with local Hugging Face models and registers hooks on the shared model: agents with inspectors or local models are served one turn at a time.
- **Servers:** both servers run each turn through `AgentPool`: on a fresh `agent.clone()` (up to `max_workers`, 8 by default) or, with one worker, on the served agent itself, loading the conversation state into it first (WhatsApp keeps it in SQLite). The pool keeps a user's turns in order and runs agent calls in its own threads; never call agents directly in async handlers. Code that runs inside agent calls (tools, `postprocess_fn`, orchestrators) can run concurrently for different users.
- **Datasets:** never modify files under dataset folders or `tests/data`; write generated dialogs elsewhere (`Dialog.to_file("...json")`).
- **Privacy:** don't persist or log user messages or personal data beyond what a feature needs; served conversations (e.g. the WhatsApp SQLite database) are personal data.
