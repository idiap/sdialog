# ChangeLog

All notable changes to SDialog will be documented here.

## [0.1.0] 2025-08-05

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

## [0.0.2] 2025-06-03

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


## [0.0.1] 2025-05-22

_(initial release)_