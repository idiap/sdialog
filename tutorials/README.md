# Sdialog Tutorials :sunglasses:
:point_right: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/idiap/sdialog/)


## Tutorials

New to SDialog? Start with the [**demo**](demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/demo.ipynb), a tour of the main features. Then pick what you came for:

### 🧪 Generate synthetic data

**Text dialogues**, in [`dialog_generation/`](dialog_generation/): tutorials 1-5 build on each other, generating dialogues for the STAR dataset in three ways, then analyzing and evaluating them.

| Tutorial | What you will learn | |
|---|---|---|
| [1. Dialogue generation with a single LLM](dialog_generation/1.single_llm_full_generation.ipynb) | Generate dialogues with a single LLM, from a description or by role-playing (use case: the STAR dataset) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/dialog_generation/1.single_llm_full_generation.ipynb) |
| [2. Multi-agent generation](dialog_generation/2.multi-agent_generation.ipynb) | Simulate dialogues between persona-driven agents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/dialog_generation/2.multi-agent_generation.ipynb) |
| [3. Multi-agent generation with orchestration](dialog_generation/3.multi-agent%2Borchestrator_generation.ipynb) | Steer agents during the conversation with orchestrators | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/dialog_generation/3.multi-agent%2Borchestrator_generation.ipynb) |
| [4. Dialogue analysis](dialog_generation/4.dialog_analysis.ipynb) | Compare the conversation flows (dialogue graphs) of the dialogues generated in tutorials 1-3 with the original STAR dialogues | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/dialog_generation/4.dialog_analysis.ipynb) |
| [5. Evaluation](dialog_generation/5.evaluation.ipynb) | Compare generated dialogues with a real dataset (STAR) using an LLM judge and metrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/dialog_generation/5.evaluation.ipynb) |

**Speech**, in [`audio/`](audio/): turn those (or any) dialogues into audio with ground-truth annotations. Requires `pip install sdialog[audio]` (a GPU is recommended).

| Tutorial | What you will learn | |
|---|---|---|
| [0. Audio generation quickstart](audio/0.quickstart.ipynb) | Turn dialogues into audio with annotations: voices, presets, reverberation, noise, telephone channels, batch generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/audio/0.quickstart.ipynb) |
| [1. Rooms and acoustics](audio/1.rooms_and_acoustics.ipynb) | Choose the room: size, RT60 or materials, microphone and speaker positions, directional microphones, a room per speaker, measured impulse responses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/audio/1.rooms_and_acoustics.ipynb) |

### 🤖 Build agents

In [`agents/`](agents/): give agents tools, reasoning and knowledge, and serve them to users.

| Tutorial | What you will learn | |
|---|---|---|
| [1. Agents with tools and thoughts](agents/1.agents_with_tools_and_thoughts.ipynb) | Build agents with hidden thinking and tool calling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/agents/1.agents_with_tools_and_thoughts.ipynb) |
| [2. RAG via tool calling](agents/2.rag_tool_integration.ipynb) | Connect an external knowledge base (RAG) to an agent through tool calling (also as a script: [`2.counsellor_rag_agent.py`](agents/2.counsellor_rag_agent.py)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/agents/2.rag_tool_integration.ipynb) |
| [3. Serving agents](agents/3.serving.ipynb) | Serve an agent through an OpenAI-compatible API, talk to it from any OpenAI client or chat app, and put it on WhatsApp (also as a script: [`whatsapp_bot.py`](agents/whatsapp_bot.py)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/agents/3.serving.ipynb) |
| [4. Testing agents with simulated users](agents/4.testing_with_simulated_users.ipynb) | Let generated users talk to your agent, check the dialogues (tool calls, LLM judges) and compare versions of the agent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/agents/4.testing_with_simulated_users.ipynb) |

### 🔬 Inspect and steer models

In [`interpretability/`](interpretability/): look inside the LLM behind an agent and change its behavior.

| Tutorial | What you will learn | |
|---|---|---|
| [1. Removing refusal with inspectors](interpretability/1.agent%2Binspector_refusal.ipynb) | Find and remove an LLM's refusal direction with inspectors and steering; needs a local Hugging Face model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/interpretability/1.agent%2Binspector_refusal.ipynb) |

---

## Running the tutorials

**On Google Colab (recommended):** click a tutorial's "Open in Colab" badge. Each notebook installs everything it needs in its first cell. For the audio and interpretability tutorials, and those using local models with Ollama, select a GPU runtime (_Runtime > Change runtime type_).

**On your own machine:** install SDialog and Jupyter, then open the notebooks from this folder:

```bash
pip install sdialog jupyter          # sdialog[audio] for the audio tutorials
jupyter notebook
```

Most tutorials use an LLM: either an API (e.g. set `OPENAI_API_KEY` for OpenAI models) or a local model with [Ollama](https://ollama.com/download) (`ollama serve`, then use a model name like `"ollama:qwen3:14b"`). The first cell of each notebook says which one it uses and how to switch.

The tutorials in `dialog_generation/` use the [STAR dataset](https://github.com/RasaHQ/STAR), which they download into `datasets/STAR` (`git clone https://github.com/RasaHQ/STAR.git datasets/STAR`).
