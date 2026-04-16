import sys
from pathlib import Path

import requests
from typing import Optional, List, Callable
from sdialog.agents import Agent, BasePersona

# Allow running this tutorial directly from the repository without requiring
# an editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def search_italian_university_data(query: str, docs_k: str = "3") -> dict:
    """
    Search the university counselling knowledge base for guidelines
    and information relevant to the student's situation.

    Use this tool whenever the conversation requires information about:
    - Universities and Department in the italian territory
    - Available resources (offices, contacts, scheduling)

    Args:
        query (str):
            A short, self-contained question or keyword phrase describing
            what information is needed.
        docs_k (str):
            Number of document snippets to retrieve (default: "3").

    Returns:
        dict: A dictionary where each key is "doc1", "doc2", … "docN" and
              the corresponding value is the text of that retrieved snippet.
              Returns {"error": "<message>"} if the retrieval service fails.
    """

    try:
        top_k = int(docs_k)
    except (ValueError, TypeError):
        top_k = 3

    endpoint = "http://localhost:7997/search"
    payload = {
        "index_name": "rag_chunks",
        "query": query,
        "top_k": top_k,
    }

    try:
        response = requests.get(endpoint, params=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}

    if isinstance(results, dict) and "documents" in results:
        items = results["documents"]
    else:
        items = results

    snippets = [item if isinstance(item, str) else item.get("text", "") for item in items]
    return {f"doc{i + 1}": text for i, text in enumerate(snippets)}


# ---------------------------------------------------------------------------
# COUNSELLOR AGENT
# ---------------------------------------------------------------------------

class Counsellor(BasePersona):
    """University Counsellor"""
    name: str = "Counsellor"

    def prompt(self) -> str:
        return """You are a university counsellor. Help the students select their courses. Answer always in english."""


agent = Agent(
    persona=Counsellor(),
    model="openai:llama3.1:8b",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="EMPTY",
    tools=[search_italian_university_data],
)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

print("\n Interactive Mode (write 'quit' to exit)\n")
while True:
    question = input("Student: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question:
        continue
    try:
        reply = agent(question)
        print(f"\nCounsellor: {reply}\n")
    except Exception as e:
        print(f"Errore: {e}\n")
