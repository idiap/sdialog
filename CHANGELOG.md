# ChangeLog

All notable changes to SDialog will be documented here.

---

## [Unreleased]

### Fixed
- **sdialog.audio**:
  - `Qwen3TTS` and `Qwen3TTSVoiceClone` no longer modify the `tts_pipeline_kwargs` dictionary they are given (e.g. `PerTurnTTS`'s `tts_kwargs`), nor their shared default one: the language, sampling and reference audio of one call were left in it.
  - `DocumentFormatNormalizer` removed newlines and tabs instead of replacing them with spaces, gluing words together (e.g. "Hello\nworld" became "Helloworld").
- **sdialog.generators**:
  - `Paraphraser.prompt()` returned `"<Template memory:...>"` instead of the instruction template.
  - `PersonaGenerator` with a persona class whose fields are required (no default): `int`, `float` and `bool` fields were never generated (nor set by `generated_attributes` rules such as `"{{20-25}}"`), keeping their `0` or `False` placeholder.
- **sdialog.interpretability**:
  - Steering functions added after attaching the inspector to the agent (e.g. `agent | inspector` and then `inspector + direction` or `steerer + inspector`, as in the documentation) were never applied.
  - `Inspector(target, agent=agent)` raised an `AttributeError`; it is now attached like `agent | inspector`.
  - Adding an inspector to an agent without a local Hugging Face model now raises a clear `RuntimeError` (it raised an `AttributeError`).
- **sdialog.orchestrators**:
  - `SimpleResponseOrchestrator` with `top_k=1`: the suggested response was split into characters ("(1) W; (2) h; ..."), or failed with an `AttributeError` when responses are given as a dictionary of actions.
- **sdialog.evaluation**:
  - `ConversationalFeatures` (and the scores based on it) and `ReadabilityScore` (and the scores based on it) now ignore the speaker names: they were computed on the "Speaker: text" lines, so `BackChannelRateScore` was always 0 and speaker names were counted as words (e.g. in `LexicalDiversityScore` or `GunningFogScore`).
  - Back-channel turns followed by punctuation (e.g. "Okay." or "Yeah!") are now detected by `BackChannelRateScore`.
  - `DialogFlowPPL` and `DialogFlowScore` (and `dialog2graph()`) ignored `ai_speaker` (`system_speaker_name`): the speaker of the first turn was always taken as the system, so the speakers were swapped in dialogues started by the user.
  - `StatsEvaluator.plot()` failed with matplotlib >= 3.11 (`boxplot()` no longer accepts `labels`).
  - Dataset evaluators given the path of a dialogue file or folder (e.g. `MeanEvaluator(score)("dialogs/")`) iterated over the characters of the path instead of loading the dialogues.
  - `DatasetComparator` with datasets given as a list (e.g. `comparator([dialogs_a, dialogs_b])`) crashed with a `TypeError` in score and embedding evaluators.
  - `ToolSequenceValidator` with `"not:"` tools: the check was inverted (e.g. with `["not:send_receipt", "charge_payment"]`, sending the receipt after the charge failed and sending it before passed). A `"not:"` tool now must not be called before the next required tool, or at all if none follows.
  - `LLMJudgePersonaAttributes` could not be created (`AttributeError`: its prompt template was rendered as a string). The persona and speaker are now passed to the prompt when judging.
  - `LLMJudgeScore.judge()` with several dialogues: the prompt asks for one score per dialogue, but the output schema only accepted a single score, so the reply always failed validation. Lists of scores (and reasons) are now accepted, each score within `[min_score, max_score]`.

### Removed
- **sdialog.audio**:
  - The legacy audio pipeline (`sdialog.audio.pipeline.to_audio` / `AudioPipeline`, dSCAPER-based), deprecated since 0.5.0, with the modules only it used: `dialog` (`AudioDialog`), `turn`, `room`, `room_generator`, `acoustics_simulator`, `room_acoustics_backends`, `impulse_response_database`, `processing`, `dscaper_utils`, `jsalt`, `legacy` and `utils`. `dialog.to_audio()` now raises a `TypeError` for the legacy pipeline's arguments (e.g. `perform_room_acoustics=True`) instead of routing them to it; use its own options (`preset=`, `reverb=`, `noise=`, `channel=`, ...) instead.
  - The `sdialog[audio-legacy]` extra, and the audio tutorials 1-10 that used the legacy pipeline (`tutorials/01_audio/0.quickstart.ipynb` covers `dialog.to_audio()`).

---

## [0.6.0] 2026-10-29 💬

### Added
- **sdialog.whatsapp** (#135, contributed by @amitbhakt):
  - `Agent.serve_whatsapp()` (and `Server.serve_whatsapp()`) to let real users chat with agents over WhatsApp through the Meta WhatsApp Cloud API. Each user (phone number) has their own conversation, saved in SQLite and restored on their next message. See [Serving Agents on WhatsApp](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#serving-agents-on-whatsapp).
  - Webhook requests are verified with the app secret (required), duplicate deliveries from Meta are ignored, and non-text messages get a short "not supported" notice.
  - Per-user rate limits: `max_messages_per_minute` (6 by default) and `max_messages_per_hour` (60 by default); extra messages are dropped without calling the LLM, with a single notice.
  - Up to `max_workers` users (8 by default) are served at the same time, each on a clone of the agent; a user's messages are always processed one at a time, in order.
  - Conversations start over when the user sends `/reset`, after `session_timeout` seconds of inactivity, or, if welcome messages are enabled on the phone number, when a user opens a chat for the first time (Meta's `request_welcome` event; the agent greets them). With the default `session_timeout="auto"`, the timeout is 30 minutes, or 24 hours if welcome messages are enabled (detected at startup); `None` disables it.
  - `session_retention_days` (30 by default): conversations of users inactive for longer are deleted from the database.
- **sdialog.agents**:
  - `Agent.get_state()` / `Agent.set_state()` to save a conversation (memory, finished flag and orchestrator state) and restore it later, e.g. on a fresh or reset agent.
  - `Agent.clone()` to copy an agent with its own conversation (memory, orchestrators, LLM settings such as the seed) that shares its language model, so weights and API clients are not duplicated.
- **sdialog.agent_pool**:
  - `AgentPool` to run up to `max_workers` conversation turns of an agent at a time (8 by default), each on a fresh clone, with turns of the same conversation kept in order. Agents with a local Hugging Face model or inspectors run one turn at a time. Used by both servers.
- **sdialog.server**:
  - `max_workers` in `Agent.serve()` and `Server.serve()` / `serve_async()` / `serve_in_thread()` to process several requests at the same time (see Changed).
  - Per-client (IP address) rate limits of chat requests: `max_requests_per_minute` (30 by default) and `max_requests_per_hour` (300 by default). Extra requests get HTTP 429 with a `Retry-After` header; pass `None` to disable a limit (e.g. for local scripts).
- **sdialog.orchestrators**:
  - `get_state()` / `set_state()` hooks for stateful orchestrators; `ChangeMindOrchestrator` saves its trigger counter.
- **sdialog.rate_limit**:
  - `RateLimiter`, a per-key sliding window rate limiter with several limits at once, used by both servers.

### Changed
- **sdialog.server**:
  - Served agents now process up to 8 requests at the same time (stateless mode, the default) instead of one, each on a clone of the agent. The agent's tools and custom functions (e.g. `postprocess_fn`) may therefore run concurrently and must be thread-safe; pass `max_workers=1` for the previous behavior. Agents with a local Hugging Face model or inspectors, and stateful agents, still process one request at a time.
  - Chat requests are now rate limited by default (30/minute and 300/hour per client IP). Scripts that call a served agent in a fast loop (e.g. evaluations) may get HTTP 429 errors; pass `max_requests_per_minute=None, max_requests_per_hour=None` to `serve()` to disable the limits.

### Deprecated
- **sdialog.audio**:
  - The legacy audio pipeline (`sdialog.audio.pipeline.to_audio` / `AudioPipeline`, dSCAPER-based), deprecated since 0.5.0, and the `sdialog[audio-legacy]` extra will be removed in an upcoming release, together with the tutorials that use them. Use `dialog.to_audio()` instead.

### Fixed
- **sdialog.server**:
  - The REST API server no longer blocks while an agent generates a reply: generation runs in a worker thread, so other requests (e.g. `/health`, `/v1/models`) are answered meanwhile instead of waiting for it.

---

## [0.5.4] 2026-10-26 🩹

### Fixed
- **sdialog.audio**:
  - `Dia2TTS` no longer crashes with a CUDA "index out of bounds" error on long generation windows, which also left the GPU process unusable (every later dialogue failed too). The reference voices took up part of Dia2's context, so long windows ran past its position limit. Generation now stops at that limit; a window cut short is regenerated one turn at a time.
  - The rotary position (RoPE) table of Dia2 is now sized to the positions generation can actually reach (`prefix_len + max_context_steps`) instead of `max_context_steps` only (#136). With reference voices, generation runs at absolute positions shifted by the prefix length, so windows were still cut short (or, before the stop above, crashed) even when they fit in the generation budget. The table grows only when needed and existing rows are unchanged, so generations that already fit give the same output. The stop above stays as a safety check against the new table size.
  - Reference voices given as audio files (e.g. from a local voice database) are now cut to `reference_seconds` (8 s by default), as arrays already were. Long reference files took up most of Dia2's context, leaving little room for the generated audio and making the crash above frequent.

---

## [0.5.0] 2026-10-15 🔊

### Added
- **sdialog.audio**:
  - New `dialog.to_audio()` engine (`sdialog.audio.synthesis`): in-memory dialogue-to-audio synthesis returning a `DialogAudio` object with the waveform and ground-truth annotations (turn segments, word timestamps, RTTM/JSON export, `save()`).
  - [Dia2](https://github.com/nari-labs/dia2) dialogue TTS (`Dia2TTS`) as the default engine, vendored under `sdialog.audio.tts.dia2` (Apache-2.0). Changes from upstream: seeding, Whisper loaded once, and in-memory reference audio.
  - Voice selection with `voices="db" | "design" | "random"`, or an explicit `{speaker: voice}` dictionary.
  - Realism controls: `pauses`, `overlap`, `reverb` (simulated rooms or measured RIRs), `noise`, `channel` (telephone codecs) and `level_jitter_db`, either shared or per speaker. Output in mono or `stereo` (one channel per speaker).
  - Scene presets: `clean`, `conversational`, `room`, `call_center`, `far_field`, `hard`, plus `register_preset()` for custom ones.
  - `dialogs_to_audio()` for parallel, multi-GPU generation, with one model per worker.
  - `BaseDialogTTS` interface for dialogue-level engines, and `PerTurnTTS` to use classic engines (Kokoro, Qwen3-TTS, ...) with `dialog.to_audio()`.
  - New `sdialog[audio-legacy]` extra with the dependencies of the legacy pipeline.

### Changed
- **sdialog.audio**:
  - `sdialog[audio]` now installs the lightweight new engine plus Dia2's dependencies (Python >= 3.10). dSCAPER, librosa, sox and the other legacy dependencies moved to `sdialog[audio-legacy]`.
  - `import sdialog.audio` no longer imports heavy dependencies; legacy names are loaded lazily.
  - The legacy helpers that lived in `sdialog/audio/__init__.py` moved to `sdialog.audio.legacy`. They can still be imported from `sdialog.audio`.

### Deprecated
- **sdialog.audio**:
  - The legacy pipeline (`sdialog.audio.pipeline.to_audio` / `AudioPipeline`, dSCAPER-based) is deprecated. `dialog.to_audio()` still routes calls that use its arguments (e.g. `perform_room_acoustics=True`) to it, with a `DeprecationWarning`.

### Fixed
- **sdialog.audio**:
  - `Dialog.to_audio()` no longer hides real import errors behind "module not installed", and the documented `perform_tts` argument no longer raises a `TypeError`.
  - Persona ages given as text (e.g. "middle-aged") no longer break voice selection.
  - `audio/prompts/*.txt` is now included in the package.
  - Removed leftover debug prints in the audio module.

---

## [0.4.7] 2026-04-14 📞

### Added
- **sdialog.audio**:
  - New telecommunication simulation engine and codecs support.
  - New modular room acoustics backend system (`pyroomacoustics` and telecommunications backends).
  - New text/audio normalization utilities in `sdialog.audio.normalizers`.
  - New prompt templates for sound-effect annotation with and without stage-aware context.
  - New tutorials for telecommunication simulation, overlaps/pauses, and sound events.

### Changed
- **sdialog.audio**:
  - Refactored acoustics simulation to a modular backend architecture.
  - Extended audio pipeline with overlap/pause control, sound-event handling, and improved post-processing.
  - Improved normalization behavior for Qwen3-TTS-based generation and voice cloning.

### Fixed
- **sdialog.generators**:
  - Improved validation error handling in `Paraphraser`.
- **sdialog**:
  - Prevented API keys from being persisted in LLM metadata.
- **sdialog.audio**:
  - Fixed sentence-transformers modality mismatch issues in audio workflows.
  - Fixed multiple robustness issues in audio processing (NumPy conversion, dScaper edge cases, dataset handling, and normalization consistency).
  - Fixed audio test/tooling compatibility issues (`torchcodec`/`qwen_tts` stubs and type hints).


---

## [0.4.6] 2026-03-12 🧠

### Added
- **sdialog.agents**:
  - New `@final_response_tool` decorator to return tool output directly as the final agent response (bypassing post-tool LLM synthesis for non-empty outputs).
  - Example:
    ```python
    from sdialog.agents import Agent, final_response_tool

    @final_response_tool
    def get_table(callsign: str) -> str:
        return "| Timestamp | Transcription |\n|---|---|\n| ... | ... |"

    agent = Agent(tools=[get_table])
    ```

### Changed
- **sdialog.evaluation**:
  - `KDEDistanceEvaluator` now supports multiple values per dialog.
  - Improved placeholder handling in STAR response generation.
  - Implemented Laplace smoothing in dialog log-likelihood computation.


---

## [0.4.5] 2026-02-18 🎙️

### Added
- **sdialog.audio**:
  - Voice cloning support in voice databases.
  - New TTS classes: `Qwen3TTS`, `Qwen3TTSVoiceClone`, and `BaseVoiceCloneTTS`.
  - Persona-based dynamic voice generation and customizable voice descriptions.
  - Improved speaker name / role mapping for voice assignment.

### Changed
- **sdialog.audio**:
  - `Dialog.to_audio()` supports output path as positional argument.
  - Audio pipeline now supports more flexible voice assignment.
  - Qwen3-TTS voice cloning set as default dynamic voice creation path.

### Fixed
- **sdialog.agents**:
  - Restored orchestration behavior when agent is served.
- **sdialog.audio**:
  - Fixed speaker-role mapping and case-sensitivity edge cases.
  - Fixed optional audio path handling in `Dialog.to_audio()`.
  - Fixed label assignment and default `keep_duplicate` handling.
- **sdialog.evaluation**:
  - Fixed division-by-zero / `log(0)` edge cases in divergence computations.
  - Added jitter handling for KL/CS divergence when variance is zero.
  - Fixed fallback behavior for unknown edges in flow score computation.


---

## [0.4.3] 2025-11-30 📊

### Added
- **sdialog.evaluation**:
  - New readability and conversational metrics.
  - New scoring classes for individual linguistic features.
  - Added `Dialog.from_folder()` to load dialogs from directories.
  - Added `ToolSequenceValidator` for validating tool usage order in dialogs.

### Changed
- **sdialog.evaluation**:
  - Enhanced plotting support with customizable titles and labels.
- **sdialog.interpretability**:
  - Improved top-k visualization and steering defaults.

### Fixed
- **sdialog.agents**:
  - Improved tool invocation error handling.
- **sdialog**:
  - Updated dialog prompt formatting.

---

## [0.4.2] 2025-11-21 🛠️

### Changed
- **sdialog.agents**:
  - Improved progress bar behavior during dialog generation.
- **sdialog.audio**:
  - Updated dependencies and dscaper integration behavior.

### Documentation
- Added clearer Kokoro TTS installation instructions.

---

## [0.4.1] 2025-11-20 🔧

### Fixed
- **sdialog.audio**:
  - Fixed dscaper timeline generation bug and test updates around deleted directories.

### Documentation
- Updated audio tutorials and README installation instructions (including sox/ffmpeg and setup guidance).

## [0.4.0] 2025-10-30 🔊

### Added
- **sdialog.audio**: New audio generation module to transform text dialogs into audio conversations
  - `Dialog.to_audio()`: Convert any text dialog to audio with a single method call
  - **Text-to-Speech (TTS)**: Support for multiple TTS engines including Kokoro and Hugging Face models
  - **Voice databases**: Automatic or manual voice assignment from Hugging Face, local storage, or on-the-fly generation based on persona attributes (age, gender, language)
  - **Acoustic simulation**: Realistic room acoustics simulation with ray tracing technology for spatial audio
  - **Microphone simulation**: Professional microphone impulse responses from brands like Shure, Sennheiser, and Sony
  - **Room generation**: Define and generate diverse room types with customizable properties (dimensions, wall materials, furniture placement)
  - **Multiple formats**: Export to WAV, MP3, or FLAC with custom sampling rates
  - **Background/foreground effects**: Add environmental sounds and acoustic variations
  - 7 comprehensive tutorials covering audio generation, acoustic simulation, rooms, voice databases, and impulse responses
- **sdialog.interpretability**: 
  - Support for inspecting layer and component inputs with `Inspector(target="model.layers.15", inspect_input=True)`
  - Support for inspecting and steering input tokens (tokens given as input, not only generated ones), e.g., `inspector.input[i][j].act` where `i` is turn index and `j` is input token index
- **sdialog.config**: 
  - Support for Anthropic backend (#100)
  - Support for Azure OpenAI backend (#100)

### Fixed
- **sdialog.agents**: Agent memory reset when no prompt system and no persona is given

---

## [0.3.3] 2025-10-30 🚀

### Added
- **sdialog.server**:
  - New module to serve agents via an Ollama/OpenAI-compatible REST API (works with UIs like Open WebUI) (#92)
- **sdialog**:
  - `Dialog.from_huggingface()` to load/download dialogues directly from Hugging Face datasets (#59)

### Changed
- **sdialog.evaluation**:
  - LLM judge methods now accept additional user-defined template arguments (e.g., like `document` in [this example](https://sdialog.readthedocs.io/en/latest/examples/index.html#example-1-yes-no-relevance-judgment-with-reasoning)) (#86)
- **sdialog.agents**:
  - Improved `Agent` initialization so agents can act as a proxy for external conversational systems (#90, fa1d8f3)

### Fixed
- **sdialog.evaluation**:
  - Corrected Flesch Reading Ease and Gunning Fog score calculations (d1d4260)


---

## [0.3.0] 2025-10-03 ✨

### Added
- **sdialog**: 
  - `Context`: new class class to explicitly model the common/shared context of conversations (#73)
  - `Dialog`: merge functionality - Added option to merge consecutive turns of the same speaker when loading a dialog (#77)
  - `Dialog`: built-in string support - Added support to built-in str functions for `Dialog` class (#83)
- **sdialog.agents**: Added new `sdialog.agents` module and moved `Agent` class inside (#81)
  - `Agent`: thinking capabilities - Agents can now handle internal thinking processes (#95)
  - `Agent`: tools support - Added tools capabilities to Agents (e.g. RAG or any other function) (#84)
    - New tutorial for agents with tools and thoughts.
- **sdialog.generators**: 
  - `ContextGenerator`: new class added to explicitly model the common/shared context of conversations (#73)
  - `Paraphraser`: new class class to paraphrase dialogues (#76)
- **sdialog.evaluation**: 
  - `LinguisticFeatureScore`: new class added to compute Flesch reading ease, Gunning fog, Hesitation rate, and/or Mean turn length (#63)
- **sdialog.personas**: 
  - `Customer` and `SupportAgent`: new personas added for customer service dialogues (#85)
  - `Persona`: Added static method to get the list of all attributes in `Persona` class (#79)


### Changed
- **sdialog**: Improved metadata handling (#66)
- **sdialog.interpretability**: Improved and simplified the way inspection targets are defined in `interpretability` submodule (#78)
- **sdialog.evaluation.base**: 
  - `LLMJudgeYesNoOutput`: Renamed attribute `yes` to `positive` (#86)
  - `LLMJudgeScoreOutput`: Renamed attribute `feedback` to `reason` (#86)

### Fixed
- **sdialog.generators**: Fixed potential bug in `PersonaDialogGenerator` class (#67)


### Enhanced
- **sdialog.agents**: Added `base_model` attribute to `Agent` to direclty access the LLM's underlying model for mechanistic interpretability (#74)
- **sdialog.config**: Added `clear_cache()` method to config (#75)

### Documentation
- API Documentation: Refactored/cleaned all components and added docstrings with examples (#82, #88)
- Updated all tutorials to work with new code and added "Open in Colab" badges
- Completed API documentation for initial official release (#87)
- Automatic generation of `llm.txt` from API documentation (24f6ee6)

---

## [0.1.0] 2025-08-05 🌱

### Added
- Multi-backend support (Hugging Face, Ollama, OpenAI, AWS)
- Enhanced persona generation (beyond initial `PersonaDialogGenerator`)
- Interpretability module (`sdialog.interpretability`): inspectors, steerers, hooks, intruders
- Evaluation module (`sdialog.evaluation`): metrics, LLM-as-a-judge scoring, evaluators, dataset comparators

### Changed
- Standardized / improved dialog format

### Notes
- >500 commits since 0.0.2 (post-JSALT 2025 consolidation)

### Pending
- Audio module (`sdialog.audio`) integration
- Documentation updates

---

## [0.0.2] 2025-06-03 🔧

### Added
- `language` attribute to `Persona` class
- `PersonaDialogGenerator` to `generators` module to support persona-based dialogue generatin with single LLM:
  ```python
  from sdialog.generators import PersonaDialogGenerator

  dialog_generator = PersonaDialogGenerator(
      model=MODEL_NAME,
      persona_a=bob_persona,
      persona_b=alice_persona,
  )

  dialog_generator.generate().print()
  ```

### Fixed
- Python 2 and 3 compatibility problem with scikit-learn (using version 0.20.1 from now on)
- PyPi: setup.py: `long_description_content_type` set to `'text/markdown'`

---

## [0.0.1] 2025-05-22 🎉

_(initial release)_