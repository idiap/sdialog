medical_dialogue_evaluator/
├── evaluators/
│   ├── __init__.py         # <-- Will contain auto-discovery logic
│   ├── base.py
│   └── clinical.py
├── __init__.py
├── config.py             # <-- NEW: Handles loading config.yaml
├── formatters.py         # <-- NEW: For flexible output (JSON, CSV, MD)
├── data_models.py
├── main_evaluator.py       # <-- Will be updated for async
├── prompts.py
├── utils.py                # <-- Will be updated to use config
└── logger.py
tests/
...
config.yaml
run_evaluation.py           # <-- Will be updated to be the async runner
README.md
