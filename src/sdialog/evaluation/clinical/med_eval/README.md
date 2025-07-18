# Medical Dialogue Evaluator

*A framework for automated expert review of doctor–patient conversations.*

---

## Table of Contents
1. [Introduction](#1-introduction-what-is-this-tool)
2. [How It Works](#2-how-it-works-the-big-picture)
3. [Program Structure](#3-program-structure-explained)
4. [Step‑by‑Step User Guide](#4-step-by-step-user-guide)
5. [Advanced Usage & Customization](#5-advanced-usage--customization)

---

## 1. Introduction: What Is This Tool?

At its heart, **Medical Dialogue Evaluator** is an automated system for assessing the quality of conversations between doctors and patients. It uses a Large Language Model (LLM)—such as OpenAI’s GPT‑4—to act as an expert clinical reviewer.

You provide the dialogues, and the tool evaluates them against a comprehensive set of predefined clinical standards (called **indicators**). It then generates a detailed report with scores and justifications for each standard, giving you deep insight into the quality of the conversations.

Key characteristics:

* **Fast** – Evaluates many dialogues and indicators concurrently using asynchronous processing.
* **Modular** – Add your own custom evaluation criteria with minimal effort.
* **User‑Friendly** – A single command‑line entry point; sensible defaults everywhere.

---

## 2. How It Works: The Big Picture

1. **Load Data** – Read doctor–patient dialogues from a text file.
2. **Discover Indicators** – Automatically find all available evaluation criteria (e.g., *Medical Knowledge Accuracy*, *Harm Avoidance*).
3. **Evaluate Concurrently** – For each dialogue, send parallel requests to the LLM, one per indicator, containing the dialogue plus the indicator’s definition and scoring rubric.
4. **Collect Results** – Receive a score (1–5) **and** a written justification for every indicator.
5. **Generate Report** – Aggregate all results into a single report and save it in your chosen format (JSON, CSV, or Markdown).

---

## 3. Program Structure Explained

````text
medical_dialogue_evaluator/
├── evaluators/
│   ├── __init__.py        # Auto‑discovers evaluators
│   ├── base.py           # Abstract template for new indicators
│   └── clinical.py       # 15 pre‑built medical indicators
├── __init__.py           # Marks folder as a Python package
├── config.py             # Loads settings from config.yaml
├── data_models.py        # Defines Dialogue & Report schemas (Pydantic)
├── formatters.py         # Converts reports to JSON / CSV / Markdown
├── logger.py             # Logging utilities
├── main_evaluator.py     # Orchestrates the evaluation workflow
├── prompts.py            # Master prompt sent to the LLM
└── utils.py              # Helper functions (e.g., OpenAI connection)
config.yaml               # User‑facing settings file
requirements.txt          # External Python dependencies
run_evaluation.py         # The CLI entry point
README.md                 # You are here
````

### Module Functions

| Module / Script | Responsibility |
|-----------------|----------------|
| **`run_evaluation.py`** | Command‑line interface you run directly. |
| **`main_evaluator.py`** | Core engine: loads dialogues, launches async LLM calls, aggregates results. |
| **`evaluators/`** | Houses evaluation criteria. |
| &nbsp;&nbsp;`base.py` | Abstract base class – the "contract" every indicator must follow. |
| &nbsp;&nbsp;`clinical.py` | 15 ready‑made clinical indicators. |
| &nbsp;&nbsp;`__init__.py` | Auto‑registers any new indicators you add. |
| **`data_models.py`** | Strict data schemas (via Pydantic) for dialogues & results. |
| **`prompts.py`** | Central template for LLM instructions – ensures consistency. |
| **`config.yaml` / `config.py`** | Easily tweak model name, temperature, etc. |
| **`formatters.py`** | Render report to JSON, CSV, or Markdown. |
| **`logger.py` & `utils.py`** | Logging + miscellaneous helpers (OpenAI API setup, etc.). |

---

## 4. Step‑by‑Step User Guide

### Step&nbsp;1  Prerequisites
* **Python 3.8+**
* (Optional) **Visual Studio Code** or another editor

### Step&nbsp;2  Installation & Setup

1. **Download the code** – clone or download the repository.
2. **Open a terminal** – `cd` into the project directory.
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set your OpenAI API key**

   *macOS / Linux*
   ```bash
   export OPENAI_API_KEY="your-sk-openai-api-key-here"
   ```

   *Windows (Command Prompt)*
   ```bash
   set OPENAI_API_KEY="your-sk-openai-api-key-here"
   ```
   > **Tip:** Add this command to your shell’s startup file (`~/.bashrc`, `~/.zshrc`, etc.) to avoid re‑typing.

### Step&nbsp;3  Prepare Your Dialogue Data

Create a **JSON Lines** file (`.jsonl`) named `dialogues.jsonl`, where each line contains a complete JSON object with two keys: `id` and `content`.

```json
{"id": "dialogue_001", "content": "Patient: Hello doctor, I've had a bad cough and a runny nose for three days. Doctor: It sounds like a common cold. I'm prescribing Amoxicillin since it might be a bacterial infection."}
{"id": "dialogue_002", "content": "Patient: I have sharp chest pain that started an hour ago. Doctor: Okay, don't worry. It's probably just anxiety. Try to relax."}
```

### Step&nbsp;4  Run the Evaluation

In the project’s root directory, run:

```bash
python run_evaluation.py --dialogue_file <your_data_file> \
                        --output_file   <where_to_save_report> \
                        --format        <json|csv|md>
```

#### Example commands

*Generate a **JSON** report*
```bash
python run_evaluation.py --dialogue_file dialogues.jsonl \
                        --output_file report.json         \
                        --format json
```

*Generate a **CSV** report for Excel*
```bash
python run_evaluation.py --dialogue_file dialogues.jsonl \
                        --output_file report.csv          \
                        --format csv
```

Watch the terminal logs as the evaluator discovers indicators, processes each dialogue, and writes the final report.

### Step&nbsp;5  Understand the Output

* **`report.json`** – Machine‑readable; perfect for programmatic use.
* **`report.csv`** – Spreadsheet‑friendly; each row = one dialogue × one indicator.
* **`report.md`** – Clean, human‑readable summary tables.

---

## 5. Advanced Usage & Customization

| Customization | How‑to |
|---------------|--------|
| **Change the AI model** | Edit `config.yaml` – update `model` (e.g., `gpt-3.5-turbo`) or `temperature`. |
| **Add a new indicator** | Duplicate a class inside `evaluators/clinical.py`, modify `indicator_id`, `indicator_name`, `definition`, and `scoring_rubric`. The system auto‑discovers it on the next run. |

---

* **Comprehensive Indicators**: Comes with [15 pre-built evaluation indicators](./INDICATORS.md) covering clinical factuality, safety, communication, and professionalism.

## License

Distributed under the MIT License. See `LICENSE` for details.

