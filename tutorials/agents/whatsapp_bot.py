"""
A support agent on WhatsApp, served with SDialog (see the "Serving Agents" tutorial, 3.serving.ipynb).

Users chat with the agent from WhatsApp, each in their own conversation. Meta sends the messages to this
server through a public HTTPS tunnel (ngrok), and the agent's replies are sent back through Meta's
WhatsApp Cloud API.

Setup (with a free Meta test number, step by step:
https://sdialog.readthedocs.io/en/latest/sdialog/index.html#serving-agents-on-whatsapp):

    pip install sdialog pyngrok
    export WHATSAPP_ACCESS_TOKEN=...      # from your Meta app (WhatsApp > API Setup)
    export WHATSAPP_PHONE_NUMBER_ID=...
    export WHATSAPP_APP_SECRET=...        # App settings > Basic
    export WHATSAPP_VERIFY_TOKEN=any-random-string
    python whatsapp_bot.py

Then set the printed tunnel URL + "/webhook" and the verify token as the callback URL in your Meta app
(WhatsApp > Configuration), subscribe to "messages", and send a message to the test number.

Conversations are saved in whatsapp_sessions.db: treat that file as personal data.
"""
import sdialog

from sdialog.agents import Agent
from sdialog.personas import SupportAgent

sdialog.config.llm("openai:gpt-4.1")  # set OPENAI_API_KEY; or a local model, e.g. "ollama:qwen3:14b"

ORDERS = {"A1001": "shipped, arriving tomorrow", "A1002": "being prepared", "A1003": "delivered on Monday"}


def get_order_status(order_id: str) -> dict:
    """Get the status of an order by its order id (e.g. "A1001")."""
    return {"order_id": order_id, "status": ORDERS.get(order_id.upper(), "not found")}


support = Agent(SupportAgent(name="Ava", product_scope="orders of an online shop",
                             communication_style="brief and friendly, as in a chat"),
                tools=[get_order_status],
                name="Support")

if __name__ == "__main__":
    support.serve_whatsapp(tunnel=True)  # credentials from the WHATSAPP_* environment variables
