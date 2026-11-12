"""
Web chat GUI for agents served with :meth:`sdialog.agents.Agent.serve` (``chat_ui=True``).

The chat is a static page (HTML, JavaScript and CSS, no build step or external resources) served by the same
FastAPI app as the REST API. Conversations are kept in the browser (``localStorage``) and sent in full with
each message, so nothing is stored on the server. The page adapts to being opened in its own tab or window, or
embedded in another page with an ``<iframe>`` or with the floating chat button of ``widget.js``.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import json
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_FILES = {"chat.js": "text/javascript", "chat.css": "text/css", "widget.js": "text/javascript"}
ROUTE_PREFIX = "sdialog_chat_ui"  # Name prefix of the chat routes, to replace them when served again
HOST_LINKS = ("new-tab", "navigate", "history")


@dataclass
class ChatUI:
    """
    Options of the web chat GUI served along with the agents' REST API.

    :param path: URL path of the chat page (``/chat`` by default), e.g. ``http://localhost:1333/chat``.
    :type path: str
    :param title: Title shown in the chat header and the browser tab (None: the agent's name, or "Chat" for
                  several agents).
    :type title: Optional[str]
    :param multiple_conversations: If True, users can keep several conversations, listed in a side panel; if
                                   False, there is a single conversation and "New chat" replaces it.
    :type multiple_conversations: bool
    :param markdown: If True, agent replies are rendered as Markdown (bold, italics, lists, headings, links and
                     code); if False, they are shown as plain text.
    :type markdown: bool
    :param show_events: If True, the agent's reasoning, orchestration instructions and tool calls are shown
                        (collapsed) above each reply. Keep it False for end users: these may contain internal
                        information, such as the instructions given to the agent or the output of its tools.
    :type show_events: bool
    :param embed_origins: Origins of the pages allowed to embed the chat in an ``<iframe>`` (e.g.
                          ``["https://example.org"]``), besides the server itself. None (default) allows any page.
    :type embed_origins: Optional[List[str]]
    :param host_links: What links to the website hosting the chat button (``widget.js``) do when clicked in the
                       chat: ``"new-tab"`` (default) opens them in a new tab, like any other link; ``"navigate"``
                       loads them in the host page; ``"history"`` shows them in the host page without reloading
                       it, for single-page apps: ``history.pushState()`` followed by a ``popstate`` event. Links
                       to other sites always open in a new tab. Pages can override it with the ``data-links``
                       attribute of the ``widget.js`` script.
    :type host_links: str

    Example:

        .. code-block:: python

            from sdialog.agents import Agent
            from sdialog.chat_ui import ChatUI

            support = Agent(name="Support")

            # Default chat GUI on http://localhost:1333/chat
            support.serve(port=1333, chat_ui=True)

            # Or with options
            support.serve(port=1333, chat_ui=ChatUI(title="Help desk", markdown=True,
                                                    embed_origins=["https://example.org"]))

            # Links to the host site (e.g. "[Show on map](/?building=42)") update the page of a single-page
            # app that embeds the chat button, instead of opening a new tab
            support.serve(port=1333, chat_ui=ChatUI(markdown=True, host_links="history"))
    """
    path: str = "/chat"
    title: Optional[str] = None
    multiple_conversations: bool = True
    markdown: bool = False
    show_events: bool = False
    embed_origins: Optional[List[str]] = None
    host_links: str = "new-tab"

    def __post_init__(self):
        self.path = "/" + self.path.strip("/")
        if self.path == "/":
            raise ValueError("ChatUI path cannot be the root path '/'")
        if isinstance(self.embed_origins, str):
            self.embed_origins = [self.embed_origins]
        if self.host_links not in HOST_LINKS:
            raise ValueError(f"ChatUI host_links must be one of {', '.join(map(repr, HOST_LINKS))}, "
                             f"not {self.host_links!r}")

    def frame_ancestors(self) -> Optional[str]:
        """Value of the CSP ``frame-ancestors`` directive (None: any page can embed the chat)."""
        if self.embed_origins is None or "*" in self.embed_origins:
            return None
        return " ".join(["'self'"] + [origin.rstrip("/") for origin in self.embed_origins])

    def page_headers(self) -> Dict[str, str]:
        """HTTP headers of the chat page: only its own files are loaded, and only listed sites can embed it."""
        csp = ("default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; "
               "connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'")
        frame_ancestors = self.frame_ancestors()
        if frame_ancestors:
            csp += f"; frame-ancestors {frame_ancestors}"
        return {"Content-Security-Policy": csp,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-cache"}

    def page_config(self, model_names: List[str]) -> Dict[str, Any]:
        """Settings passed to the chat page's script."""
        agents = [{"id": name, "name": display_name(name)} for name in model_names]
        title = self.title or (agents[0]["name"] if len(agents) == 1 else "Chat")
        return {"title": title,
                "agents": agents,
                "apiUrl": f"{self.path}/api/reply",
                "storageKey": f"sdialog-chat:{self.path}",
                "multipleConversations": self.multiple_conversations,
                "markdown": self.markdown,
                "showEvents": self.show_events,
                "hostLinks": self.host_links}

    def render_page(self, model_names: List[str]) -> str:
        """Return the chat page's HTML, with its settings embedded as JSON."""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        config = self.page_config(model_names)
        # "<" escaped so that no value can close the <script> element holding the settings
        config_json = json.dumps(config, ensure_ascii=False).replace("<", "\\u003c")
        values = {"config": config_json, "title": _escape_html(config["title"]), "base": _escape_html(self.path)}
        # Single pass, so that placeholders inside the values (e.g. a title "{{base}}") are left as they are
        return re.sub(r"\{\{(config|title|base)\}\}", lambda match: values[match.group(1)], html)

    def render_widget(self) -> str:
        """Return the chat button's script (``widget.js``), with the server's settings."""
        script = (STATIC_DIR / "widget.js").read_text(encoding="utf-8")
        return script.replace('"{{host_links}}"', json.dumps(self.host_links))


def display_name(model_name: str) -> str:
    """Agent name shown in the chat for a served model name (without the default ``:latest`` tag)."""
    return model_name[:-len(":latest")] if model_name.endswith(":latest") else model_name


def events_to_ui(events: List) -> List[Dict[str, Any]]:
    """
    Convert the events of an agent turn into what the chat shows above the reply (reasoning, orchestration
    instructions and tool calls with their output).
    """
    outputs = {}
    for event in events:
        if event.action == "tool" and event.actionLabel == "output" and isinstance(event.content, dict):
            outputs[event.content.get("call_id")] = event.content.get("output")

    ui_events = []
    for event in events:
        if event.action == "think":
            ui_events.append({"type": "think", "label": "Reasoning", "content": _to_text(event.content)})
        elif event.action.startswith("instruct"):
            scope = "persistent" if "persist" in event.action else "one-time"
            label = f"Instruction ({event.actionLabel}, {scope})" if event.actionLabel else f"Instruction ({scope})"
            ui_events.append({"type": "instruction", "label": label, "content": _to_text(event.content)})
        elif event.action == "tool" and event.actionLabel == "call":
            call = event.content if isinstance(event.content, dict) else {}
            name = call.get("name", "unknown")
            ui_events.append({"type": "tool", "label": f"Tool: {name}",
                              "content": f"Arguments: {_to_text(call.get('args', {}))}\n"
                                         f"Output: {_to_text(outputs.get(call.get('id'), '(no output)'))}"})
    return ui_events


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _escape_html(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))


def remove_routes(app) -> None:
    """
    Remove the chat GUI routes from the server's FastAPI app, if any.

    :meta private:
    """
    app.router.routes = [route for route in app.router.routes
                         if not (getattr(route, "name", None) or "").startswith(ROUTE_PREFIX)]


def register_routes(app, server, chat_ui: ChatUI) -> None:
    """
    Add the chat GUI routes to the server's FastAPI app (replacing those of a previous ``serve()`` call).

    :param app: The FastAPI app of the server.
    :param server: The :class:`~sdialog.server.Server` class (agents, rate limits and agent pools).
    :param chat_ui: Options of the chat GUI.
    :type chat_ui: ChatUI

    :meta private:
    """
    from fastapi import Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from pydantic import BaseModel, Field

    from ..server import ChatCompletionRequest, ChatMessage

    class ChatUIRequest(BaseModel):
        model: str = Field(..., description="Model name of the agent to talk to")
        messages: List[ChatMessage] = Field(..., description="Conversation so far, ending with the user message")

    remove_routes(app)
    base = chat_ui.path

    async def chat_page():
        return HTMLResponse(chat_ui.render_page(server.list_agents()), headers=chat_ui.page_headers())

    def static_file(name: str):
        if name == "widget.js":
            content = chat_ui.render_widget().encode("utf-8")
        else:
            content = (STATIC_DIR / name).read_bytes()

        async def get_file():
            # The widget script is loaded by other sites, so it is also served to cross-origin requests
            return Response(content, media_type=STATIC_FILES[name],
                            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})
        return get_file

    async def reply(request: ChatUIRequest, raw_request: Request):
        limited = server._rate_limit_response(raw_request, ollama=True)  # Plain {"error": message} body
        if limited:
            return limited
        if request.model not in server._agents:
            return JSONResponse(status_code=404, content={"error": f"Agent '{display_name(request.model)}' "
                                                                   "is not available."})
        messages = [m for m in request.messages if m.role in ("user", "assistant")]
        if not messages or messages[-1].role != "user":
            return JSONResponse(status_code=400, content={"error": "The last message must be a user message."})

        try:
            events = await server._generate_events(ChatCompletionRequest(model=request.model, messages=messages))
        except Exception as e:
            logger.error(f"Error generating the chat reply: {e}")
            return JSONResponse(status_code=500, content={"error": "The agent could not reply, please try again."})

        utterances = [e for e in events if e.action == "utter"]
        content = {"reply": utterances[-1].content if utterances else ""}
        if chat_ui.show_events:
            content["events"] = events_to_ui(events)
        return JSONResponse(content=content)

    app.add_api_route(base, chat_page, methods=["GET"], name=f"{ROUTE_PREFIX}_page", include_in_schema=False)
    for name in STATIC_FILES:
        app.add_api_route(f"{base}/{name}", static_file(name), methods=["GET"],
                          name=f"{ROUTE_PREFIX}_{name}", include_in_schema=False)
    app.add_api_route(f"{base}/api/reply", reply, methods=["POST"], name=f"{ROUTE_PREFIX}_reply",
                      include_in_schema=False)
