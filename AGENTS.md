<!--
AGENTS.md — Machine-focused operational guide for the sdialog repository.
This file is intentionally concise, action‑oriented, and complementary to README.md.
Agents: Prefer following explicit commands / conventions here over guessing.
Human contributors: See README.md + CONTRIBUTING.md for narrative context.
-->

# AGENTS Guidelines for This Repository

Synthetic Dialogue Generation, Orchestration, Evaluation, Interpretability.

Focus for agents:
1. Reproducible environment & dependency handling
2. Safe model / API usage & configuration overrides
3. Standard commands (build, test, lint, docs, packaging)
4. Dataset & artifacts locations
5. Extension points (personas, orchestrators, evaluators, inspectors)
6. Performance & caching knobs
7. Contribution / PR hygiene
8. Security / privacy considerations

If an instruction here conflicts with user chat input, defer to the user. For file‑local changes prefer editing minimal regions.

---

## 1. Repository Layout (key paths only)

```
sdialog/                Core library (packaged via pyproject.toml)
  requirements.txt      Runtime + dev dependencies (dynamic in pyproject)
  src/ or package root  (Package modules live directly under sdialog/)
  tutorials/            Example notebooks & advanced usage
  docs/                 Sphinx docs (ReadTheDocs)
  AGENTS.md             (This file)
  README.md             Human overview
  CONTRIBUTING.md       Contribution guidelines
  LICENSE               MIT
datasets/ | Datasets/   External dialogue / STAR dataset snapshots (read-only)
AutoTOD/, AgenTOD/, task-oriented-dialogue/  Related research / comparative tooling (not installed by default)
JSALT/                  Workshop materials & experiments
```

Only the `sdialog` Python package is published to PyPI. Other folders are experimental/supporting and may have separate requirements.

---

## 2. Environment Setup

Supported Python: >=3.9 (see `pyproject.toml`). Recommended fresh virtual environment.

### Quick install (library usage)
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]  # if an extras block is later added; otherwise:
pip install -r sdialog/requirements.txt
pip install -e sdialog
```

### Clean reinstall
```
pip uninstall -y sdialog || true
pip install -e sdialog
```

### Dependency resolution notes
* `pyproject.toml` uses `setuptools.dynamic` to read `requirements.txt`.
* Pin additions: modify `sdialog/requirements.txt` then reference in PR.
* Avoid silently upgrading core ML libs (`torch`, `transformers`) without noting compatibility.
* If adding optional backend (OpenAI / Ollama / AWS / Google), ensure minimal import‑time cost; guard imports.

### GPU / Torch
* Do not auto‑install CUDA wheels; leave to user environment.
* If a test requires GPU, skip gracefully when `torch.cuda.is_available()` is False.

---

## 3. LLM Configuration & Global State

Central API:
```
import sdialog
sdialog.config.llm("provider:model", temperature=0.7, top_p=0.9)
sdialog.config.llm_params(max_tokens=512)
```
Providers use prefix naming:
* `openai:MODEL`
* `huggingface:REPO`
* `ollama:MODEL`
* `amazon:bedrock-model-id`
* `google:genai-model-id`

When modifying code that instantiates models:
* Always allow explicit `model=` override in class constructors.
* Keep default fallback: `config["llm"]["model"]`.
* Avoid hard‑coding API keys; rely on environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Never commit keys.

Think segments: If `think=True`, internal prompts may contain `<think>...</think>` sections; pattern customizable via `thinking_pattern`.

Tools: Agents accept simple Python callables; treat them as pure (side‑effect‑light) unless clearly documented.

---

## 4. Core Command Cheat‑Sheet

| Task | Command |
|------|---------|
| Install dev deps | `pip install -r sdialog/requirements.txt` |
| Editable install | `pip install -e sdialog` |
| Lint (flake8) | `flake8 sdialog` |
| Run tests (pytest) | `pytest -q` |
| Coverage | `pytest --cov=sdialog --cov-report=term-missing` |
| Build docs | `pip install -r sdialog/docs/requirements.txt && (cd sdialog && make -C docs html || sphinx-build -b html docs docs/_build/html)` |
| Package sdist/wheel | `python -m build` (if build backend tooling added) |
| Update version | Edit `sdialog/util/__init__.py` (where `__version__` lives) |
| Format tables | Use `sdialog.util.dict_to_table(..., markdown=True)` |

Tests assume importable `sdialog`; ensure editable install before running.

---

## 5. Data & Artifacts

STAR dataset utilities live under `sdialog/datasets`. External raw STAR data likely mirrored under `Datasets/STAR` or `datasets/STAR`.

Agent operations MUST NOT mutate dataset source files. For synthetic generation:
* Write outputs (dialogs JSON) into a new folder (e.g., `outputs/` or `results/`).
* Use `Dialog.to_file()` for serialization; prefer `.json` extension.

Large artifacts (embeddings, cached evaluations) should be placed in a git‑ignored path (e.g., `cache/`, configurable via `sdialog.config.set_cache(path, enable=True)`).

---

## 6. Personas, Agents, Orchestrators (Extension Points)

Subclassing patterns:
* Personas: Inherit from `BaseAttributeModel` (see `sdialog.personas`). Keep field names snake_case.
* Orchestrators: Inherit `BaseOrchestrator` or `BasePersistentOrchestrator`; implement `instruct(self, dialog, utterance)`.
* Generators: Extend `BaseAttributeModelGenerator` for new structured generation flows.
* Evaluators / Judges: Inherit `BaseDialogScore`, `BaseDialogFlowScore`, `BaseLLMJudge`, or dataset evaluator bases.
* Interpretability: Implement new `Steerer` or extend `Inspector` logic carefully—avoid heavy work inside hook functions.

Composition syntax: `agent = agent | orchestrator_or_inspector` returns a cloned agent with the component attached.

When adding new components:
1. Provide docstring with Example section.
2. Ensure `.json()` method returns serializable config.
3. Respect persona/context immutability unless explicitly cloning.

---

## 7. Dialogue Generation Workflow (Minimal)

```
from sdialog.personas import Persona
from sdialog.agents import Agent
from sdialog import Context

user = Agent(persona=Persona(name="User", role="seeker"), first_utterance="Hi")
bot  = Agent(persona=Persona(name="Bot", role="helper"))
ctx  = Context(location="lab", topics=["safety"])
dialog = user.dialog_with(bot, context=ctx, max_turns=20)
dialog.print(orchestration=True)
dialog.to_file("example_dialog.json")
```

Common variations:
* Use `PersonaGenerator` / `ContextGenerator` for automatic diversification.
* Apply `Paraphraser` to augment textual style while preserving semantics.
* Prepend orchestrators for flow constraints (length, reflex, suggestion, change‑of‑mind).

---

## 8. Evaluation & Comparison

Scoring guidelines:
* For batch evaluation prefer building a list of `Dialog` objects, then pass into `DatasetComparator`.
* When creating new metrics: implement `.score(dialog)` returning primitive numeric or structured result; keep deterministic for identical input.
* LLM judges should expose `feedback` toggle to control cost.

Edge cases to handle in metrics:
* Empty dialogue (return None or 0 with documented behavior)
* Single turn (avoid division by zero in transition metrics)
* Non‑ASCII text (ensure `.lower()` safe)

---

## 9. Interpretability & Steering

Inspector usage requires model objects exposing accessible layer names. When attaching hooks:
* Use precise target strings (e.g., `model.layers.5.post_attention_layernorm`).
* Avoid capturing every layer unless necessary—memory blowup risk.
* Steering intervals `(start, end)` reduce overhead.
* Remove or clear inspectors after experiments to avoid stale references (`agent.clear_inspectors()`).

Persisted analyses: Store extracted activations / summaries under a non‑tracked directory.

---

## 10. Caching & Reproducibility

Enable caching explicitly:
```
import sdialog
sdialog.config.set_cache("./cache", enable=True)
```
* Clear cache between benchmark runs: `sdialog.config.clear_cache()`.
* Set seeds when generating: pass `seed=` to generators / dialog methods.
* Persona / context generators support rule templates; always log the chosen rules for reproducibility.

---

## 11. Code Style & Linting

* Max line length: 120 (flake8 config)
* Prefer explicit imports (`from sdialog.personas import Persona`) over wildcard.
* Type hints: Add when public API surface is extended; avoid breaking backward compatibility.
* Keep pure data classes (pydantic) side‑effect free in `__init__`.
* Avoid heavy network I/O during module import—delay until method invocation.

---

## 12. Testing Guidelines

* Use `pytest` naming: `test_*.py`.
* Scope: Unit tests for persona cloning, orchestration logic, evaluation metrics; integration tests for end‑to‑end generation with a mock / lightweight model.
* Avoid live external API calls in default test suite—mock LLM responses or set environment variable gate (e.g., `SDIALOG_RUN_LIVE=1`).
* For stochastic processes, pass a fixed `seed` and assert structural properties instead of exact text.
* Ensure new orchestrators have at least one deterministic trigger test.

---

## 13. PR / Commit Instructions

Before opening PR:
1. Run `flake8 sdialog`.
2. Run `pytest -q` (or subset if large).
3. Update / add tests for any new public method or class.
4. Update docs / README / tutorials if API additions are user‑visible.
5. If adding dependency: justify in PR description; prefer lightweight alternatives.
6. If modifying persona or dialog schema fields: ensure backward compatibility, update serialization tests.

Commit message format (recommendation):
```
[sdialog] <short imperative summary>

Optional body explaining rationale, tradeoffs, migration notes.
```

Version bumps: maintainers update `__version__`; do not bundle unrelated refactors with release commits.

---

## 14. Security / Privacy Considerations

* Never log raw API keys or secrets.
* Avoid persisting full model outputs that may contain user-provided PII unless purposefully anonymized.
* When steering / hooking, do not serialize raw activation tensors in public artifacts; aggregate or hash if sharing.
* Validate file paths from user input to prevent directory traversal in any future file‑loading utilities.

---

## 15. Performance Tips

* Minimize repeated tokenizer/model loads: reuse configured global model when possible.
* For large batch generation: lower `temperature`, set `max_tokens`, and disable `think` unless needed.
* Use `Paraphraser(turn_by_turn=True)` only for small dialogs; otherwise batch.
* Downstream embedding evaluations: reuse a single `SentenceTransformerDialogEmbedder` instance.

---

## 16. Common Pitfalls & Resolutions

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| Empty orchestrator output | Condition never true | Add logging / test condition with sample utterance |
| High memory usage | Too many inspector hooks | Narrow layer targets or detach inspector |
| Non‑deterministic tests | Missing seed or cached LLM randomness | Pass `seed=` and disable cache |
| Import error for backend | Optional dependency missing | Add to `requirements.txt` or guard import |
| Slow first call | Model warmup / remote latency | Document one warmup call in benchmark scripts |

---

## 17. Adding a New Orchestrator (Template)

```
from sdialog.orchestrators import BaseOrchestrator

class MyKeywordOrchestrator(BaseOrchestrator):
	def __init__(self, keyword: str, instruction: str):
		super().__init__()
		self.keyword = keyword.lower()
		self.instruction = instruction

	def instruct(self, dialog, utterance: str):
		if utterance and self.keyword in utterance.lower():
			return self.instruction
		return None
```
* Add test verifying instruction appears when keyword present.
* Document usage in docstring.

---

## 18. Automated Parsing Hints (for coding agents)

Heuristics you may apply:
* For generation tasks, search for `dialog_with(` or `PersonaGenerator(` usage examples to scaffold new scripts.
* For evaluation additions, locate subclasses of `BaseDialogScore` to pattern‑match.
* Use `attributes()` static method on persona classes to programmatically list fields.

### LLM-friendly documentation
* This project provides an `llm.txt` file at https://sdialog.readthedocs.io/en/latest/llm.txt following the [llms.txt specification](https://llmstxt.org/).
* Agents can fetch this file for structured, curated information about the project with: `#fetch https://sdialog.readthedocs.io/en/latest/llm.txt`
* The llm.txt contains project summary, key documentation links, and API references optimized for LLM consumption.

---

## 19. Updating Documentation

* API docs auto-built from docstrings via Sphinx. Keep docstrings concise, with Example blocks.
* When adding config keys, update narrative docs (section: Configuration & Control) if present.
* If adding new tutorial notebook: place under `sdialog/tutorials/` and keep outputs cleared.

---

## 20. De‑scoping / Removal Policy

When removing an experimental module:
1. Mark as deprecated in previous release (docstring + warnings).
2. Remove only after one minor version or if broken beyond trivial repair.
3. Provide migration hint in changelog.

---

## 21. Changelog Discipline

* Keep `CHANGELOG.md` (if present) or GitHub Releases notes updated for: breaking changes, new components, deprecations.
* Use semantic-ish grouping: Added / Changed / Fixed / Deprecated / Removed / Security.

---

## 22. Final Notes for Agents

* Prefer minimal diffs: isolate logical change sets.
* Run tests after modifying orchestration, evaluators, or persona schema.
* If uncertain about a field meaning, inspect docstring (source of truth) rather than guessing.
* Do not auto‑reformat entire files unless style violations present.

End of AGENTS.md

