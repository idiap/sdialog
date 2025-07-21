# Medical Dialogue Evaluator: A Deep Dive into the Program Structure

This guide provides a detailed look at the internal structure of the Medical Dialogue Evaluation Framework. It explains the purpose of each file and directory, helping you understand how the system works and how you can confidently customize it.

---

## Table of Contents
- [1. The Core Concept](#1-the-core-concept)
- [2. High-Level Workflow](#2-high-level-workflow)
- [3. Directory and File Structure](#3-directory-and-file-structure)
- [4. Module-by-Module Breakdown](#4-module-by-module-breakdown)
  - [The Main Script (`run_evaluation.py`)](#the-main-script-run_evaluationpy)
  - [The Core Package (`medical_dialogue_evaluator/`)](#the-core-package-medical_dialogue_evaluator)
    - [`main_evaluator.py`: The Engine](#main_evaluatorpy-the-engine)
    - [`evaluators/`: The Knowledge Base](#evaluators-the-knowledge-base)
    - [`data_models.py`: The Data Blueprint](#data_modelspy-the-data-blueprint)
    - [`prompts.py`: The LLM's Instruction Manual](#promptspy-the-llms-instruction-manual)
    - [`formatters.py`: The Report Generator](#formatterspy-the-report-generator)
    - [`config.py` & `config.yaml`: The Settings Panel](#configpy--configyaml-the-settings-panel)
    - [`utils.py`: The Toolbox](#utilspy-the-toolbox)
    - [`logger.py`: The Diary](#loggerpy-the-diary)
- [5. How It All Works Together](#5-how-it-all-works-together)

---

## 1. The Core Concept

The framework is designed around a simple idea: **separation of concerns**. Each part of the program has one specific job. This makes the code clean, easy to understand, and safe to modify. For example, the code that defines the evaluation criteria is completely separate from the code that generates plots.

The key components are:
-   **A Command-Line Interface (CLI)** to start the process.
-   **A Core Evaluation Engine** that manages the workflow.
-   **A Modular Set of "Evaluators"** that define *what* to measure.
-   **Helper Modules** for tasks like data formatting, configuration, and connecting to the AI.

---
## 2. High-Level Workflow

When you run the program, here's what happens behind the scenes:

1.  The **`run_evaluation.py`** script starts up, parses your command-line arguments (like file paths and output format), and loads your dialogue data.
2.  It then calls the **`main_evaluator.py`** module, which is the heart of the application.
3.  The `main_evaluator` uses the logic in **`evaluators/__init__.py`** to automatically discover all the available clinical indicators.
4.  For each dialogue, it creates a list of asynchronous tasks—one for each indicator.
5.  It sends all these tasks to the OpenAI API at once. Each request uses the master template from **`prompts.py`** filled with the specific indicator's details.
6.  As the AI responds, the `main_evaluator` collects the structured results, which are validated against the schemas in **`data_models.py`**.
7.  Once all evaluations are complete, the `run_evaluation.py` script takes over again. It uses a function from **`formatters.py`** to convert the results into your desired format (JSON, CSV, or Markdown).
8.  If requested, it also calls the `.plot()` method from the `data_models.py` file to generate and save visualizations for each report.

---
## 3. Directory and File Structure

The project is organized logically to separate the user-facing scripts from the core application logic.

├── medical_dialogue_evaluator/  # The core Python package
│   ├── evaluators/              # Sub-package for all indicator logic
│   │   ├── init.py
│   │   ├── base.py
│   │   └── clinical.py
│   ├── init.py
│   ├── config.py
│   ├── data_models.py
│   ├── formatters.py
│   ├── logger.py
│   ├── main_evaluator.py
│   ├── prompts.py
│   └── utils.py
├── config.yaml                  # User-editable settings
├── requirements.txt             # Project dependencies
├── run_evaluation.py            # The script you run
└── README.md                    # Main project documentation

---
## 4. Module-by-Module Breakdown

### The Main Script (`run_evaluation.py`)
-   **Purpose**: This is the **user's entry point**. It's the only file you need to interact with from your terminal.
-   **Key Functions**:
    -   Uses `argparse` to define and read command-line arguments (`--dialogue_file`, `--format`, `--plot`, etc.).
    -   Loads the dialogue data from your `.jsonl` file.
    -   Initializes and runs the main `DialogueEvaluator`.
    -   Calls the appropriate formatter to write the output file.
    -   Triggers the plot generation if requested.

### The Core Package (`medical_dialogue_evaluator/`)

This directory contains all the application's internal logic.

#### `main_evaluator.py`: The Engine
-   **Purpose**: To **orchestrate the entire evaluation process**.
-   **Key Functions**:
    -   The `DialogueEvaluator` class is the main workhorse.
    -   It takes a list of dialogues and a list of evaluators.
    -   It uses `asyncio` to manage concurrent API calls to the LLM, making the process fast and efficient.
    -   It handles errors gracefully, ensuring that a failure in one evaluation doesn't stop the entire batch.

#### `evaluators/`: The Knowledge Base
This sub-package defines *what* to evaluate.
-   `base.py`: Defines the **template** or "contract" for all evaluators. The `BaseEvaluator` abstract class ensures that any new indicator you create will have the required attributes (`indicator_id`, `name`, etc.), making it compatible with the rest of the system.
-   `clinical.py`: This is the **library of indicators**. It contains the 15 pre-built classes, each representing a specific clinical standard. Each class provides the concrete details (definition, scoring examples) for its indicator. This is the primary file you would edit to add or modify indicators.
-   `__init__.py`: This file makes the system **modular and extensible**. Its `discover_evaluators()` function automatically scans the `clinical.py` file and finds any class that follows the `BaseEvaluator` template. This means you never have to manually update a list when you add a new indicator.

#### `data_models.py`: The Data Blueprint
-   **Purpose**: To define the **structure and rules for our data**.
-   **Key Functions**:
    -   Uses the `Pydantic` library to create strict data schemas (`Dialogue`, `EvaluationResult`, `FullEvaluationReport`).
    -   This guarantees that data flowing through the application is always in the correct format, preventing bugs.
    -   It also contains the powerful `.plot()` method, which adds visualization capabilities directly to the report objects.

#### `prompts.py`: The LLM's Instruction Manual
-   **Purpose**: To hold the **master prompt template**.
-   **Key Functions**:
    -   Contains a single, well-engineered prompt string with placeholders (e.g., `{{ indicator_name }}`).
    -   Centralizing the prompt ensures consistency across all evaluations and makes it easy to experiment with different instructions for the LLM without changing the core Python logic.

#### `formatters.py`: The Report Generator
-   **Purpose**: To **convert the final results into different file formats**.
-   **Key Functions**:
    -   Provides separate functions (`to_json`, `to_csv`, `to_markdown`) for each output type.
    -   This keeps the formatting logic separate from the evaluation logic, making it easy to add new output formats in the future.

#### `config.py` & `config.yaml`: The Settings Panel
-   **Purpose**: To manage **user-configurable settings**.
-   **Key Functions**:
    -   `config.yaml` is a simple text file where users can easily change parameters like the LLM model name without touching the code.
    -   `config.py` safely loads the values from the YAML file, provides default settings if the file is missing, and makes these settings available to the rest of the application.

#### `utils.py`: The Toolbox
-   **Purpose**: To contain **reusable helper functions**.
-   **Key Functions**:
    -   `get_llm_client()`: This function handles the technical details of setting up and authenticating the connection to the OpenAI API, using the settings loaded from `config.py`.
    -   `EvaluationOutput`: Defines the Pydantic model that `langchain` uses to parse the LLM's JSON response.

#### `logger.py`: The Diary
-   **Purpose**: To provide **structured status updates and error messages**.
-   **Key Functions**:
    -   Sets up a standardized logger that prints informative messages to the terminal (e.g., "Initializing evaluator...", "Saving report to...").
    -   This is far more robust than using simple `print()` statements and is essential for debugging.

---
## 5. How It All Works Together

When you execute `python run_evaluation.py ...`, you are kicking off a chain reaction:

1.  `run_evaluation.py` reads your command and data.
2.  It passes control to the `DialogueEvaluator` in `main_evaluator.py`.
3.  The evaluator discovers the indicators from `evaluators/clinical.py`.
4.  It loops through your dialogues, sending off batches of asynchronous API requests. Each request is a prompt from `prompts.py` filled with data.
5.  The LLM's responses are collected and validated against the schemas in `data_models.py`.
6.  The final, structured report is passed back to `run_evaluation.py`.
7.  `run_evaluation.py` uses a function from `formatters.py` to write the report to a file and calls the `.plot()` method from `data_models.py` to create the charts.

By keeping each module focused on a single task, the framework remains powerful, flexible, and easy to maintain.

