"""
OpenAI-compatible RESTful API server for SDialog agents.

This module provides a Server class that serves SDialog agents as OpenAI-compatible
chat completion endpoints, enabling integration with OpenAI-compatible clients
like Open WebUI.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import json
import uuid
import time
import asyncio
import logging

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from threading import Lock

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI dependencies are required to run the server. "
        "Please install them with: pip install fastapi uvicorn"
    )

from .agents import Agent

logger = logging.getLogger(__name__)

_role2msg_class = {
    "user": HumanMessage,
    "system": SystemMessage,
    "assistant": AIMessage
}


class ChatMessage(BaseModel):
    """
    OpenAI-compatible chat message.

    :meta private:
    """
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the speaker")


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    :meta private:
    """
    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")


class OllamaChatRequest(BaseModel):
    """
    Ollama-compatible chat request.

    :meta private:
    """
    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    stream: Optional[bool] = Field(True, description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options")


class OllamaChatResponse(BaseModel):
    """
    Ollama-compatible chat response.

    :meta private:
    """
    model: str = Field(..., description="Model used")
    created_at: str = Field(..., description="Creation timestamp")
    message: ChatMessage = Field(..., description="Response message")
    done: bool = Field(True, description="Whether the response is complete")


class OllamaChatResponseNonStreaming(BaseModel):
    """
    Ollama-compatible chat non-streaming response.

    :meta private:
    """
    model: str = Field(..., description="Model used")
    created_at: str = Field(..., description="Creation timestamp")
    message: ChatMessage = Field(..., description="Response message")


class ChatCompletionChoice(BaseModel):
    """
    OpenAI-compatible choice in chat completion response.

    :meta private:
    """
    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.

    :meta private:
    """
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")


class Server:
    """
    Static server class for serving SDialog agents as OpenAI-compatible API.

    This server provides OpenAI-compatible chat completion endpoints that can be used
    with clients like Open WebUI. The server handles agent memory internally,
    only processing new messages while maintaining conversation context.
    """

    _agents: Dict[str, Agent] = {}
    _agent_locks: Dict[str, Lock] = {}
    _app: Optional[FastAPI] = None
    _stateless: bool = False

    @classmethod
    def _setup_agents(cls,
                      agents: Union[Agent, List[Agent]],
                      model_names: Optional[Union[str, List[str]]] = None,
                      stateless: bool = None) -> str:
        """
        Set up the agent for serving, including model name processing and FastAPI app creation.

        :param agents: The SDialog agent or a list of agents to serve.
        :type agents: Union[Agent, List[Agent]]
        :param model_names: Model names to expose in the API (defaults to agent's name).
        :type model_names: Optional[Union[str, List[str]]]
        :param stateless: If True, the agent will not maintain memory between requests.
        :type stateless: bool
        :return: The processed model name.
        :rtype: str
        """
        if not isinstance(agents, list):
            agents = [agents]
        if model_names is not None and not isinstance(model_names, list):
            model_names = [model_names]
        if model_names is not None and len(model_names) != len(agents):
            raise ValueError("Length of model_name list must match length of agent list")

        model_name = None
        for ix, agent in enumerate(agents):

            if model_names is None:
                model_name = getattr(agent, 'name', f'sdialog-agent-{len(cls._agents)}')
            else:
                model_name = model_names[ix]

            if ":" not in model_name:
                model_name = f"{model_name}:latest"

            # Register the agent
            agent._can_finish = False  # Disable internal end-of-dialog detection; turn-taking is client-driven
            cls._agents[model_name] = agent
            cls._agent_locks[model_name] = Lock()

        if stateless is not None:
            cls._stateless = stateless

        # Create FastAPI app if not exists
        if cls._app is None:
            cls._create_app()

        return model_name

    @classmethod
    def serve(cls,
              agents: Union[Agent, List[Agent]],
              host: str = "0.0.0.0",
              port: int = 1333,
              stateless: bool = True,
              model_names: Optional[Union[str, List[str]]] = None,
              log_level: str = "info") -> None:
        """
        Serve SDialog agents as an OpenAI-compatible RESTful API.

        This method automatically detects the environment and chooses the appropriate
        server startup method. In standard environments (command line, scripts), it
        uses uvicorn.run(). In Jupyter notebooks or other environments with existing
        event loops, it automatically falls back to a threaded server.

        :param agents: The SDialog agent or a list of agents to serve.
        :type agents: Union[Agent, List[Agent]]
        :param host: Host address to bind the server to.
        :type host: str
        :param port: Port number to bind the server to.
        :type port: int
        :param stateless: If True, the agent will not maintain memory between requests and the
                          full context must be provided with each request.
        :type stateless: bool
        :param model_names: Model names to expose in the API (defaults to agent's name).
        :type model_names: Optional[Union[str, List[str]]]
        :param log_level: Logging level for the server.
        :type log_level: str


        Example:

            .. code-block:: python

                from sdialog import Persona
                from sdialog.agents import Agent
                from sdialog.server import Server

                # Create two agents
                user = Agent(persona=Persona(name="Dr. Nebula", role="Astrobotanist seeking alien spores"),
                             name="Scientist")
                bot = Agent(persona=Persona(name="StationCore", role="Sarcastic habitat control AI"),
                            name="Bot")

                # Serve them as an OpenAI-compatible API
                Server.serve([user, bot], port=1333)
                # Output:
                # Starting server for agents on localhost:1333
                # > 2 registered agents: Scientist:latest, Bot:latest
        """
        cls._setup_agents(agents, model_names, stateless)

        logger.info(f"Starting server for agents on {host}:{port}")
        logger.info(f"> {len(cls.list_agents())} registered agents: {', '.join(cls.list_agents())}")

        try:
            uvicorn.run(cls._app, host=host, port=port, log_level=log_level)
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                logger.info("Detected existing event loop (likely Jupyter environment). "
                            "Falling back to threaded server...")
                # Use the threaded version as fallback
                # Agents were already added in _setup_agents
                return cls.serve_in_thread([], host, port, stateless, None, log_level)
            else:
                # Re-raise if it's a different RuntimeError
                raise

    @classmethod
    async def serve_async(cls,
                          agents: Union[Agent, List[Agent]],
                          host: str = "0.0.0.0",
                          port: int = 1333,
                          stateless: bool = True,
                          model_names: Optional[Union[str, List[str]]] = None,
                          log_level: str = "info") -> None:
        """
        Serve SDialog agents as an OpenAI-compatible RESTful API (async version).

        This method is designed for use in environments with existing event loops,
        such as Jupyter notebooks, where uvicorn.run() would fail.

        :param agents: The SDialog agent or a list of agents to serve.
        :type agents: Union[Agent, List[Agent]]
        :param host: Host address to bind the server to.
        :type host: str
        :param port: Port number to bind the server to.
        :type port: int
        :param stateless: If True, the agent will not maintain memory between requests and the
                          full context must be provided with each request.
        :type stateless: bool
        :param model_names: Model names to expose in the API (defaults to agent's name).
        :type model_names: Optional[Union[str, List[str]]]
        :param log_level: Logging level for the server.
        :type log_level: str
        """
        cls._setup_agents(agents, model_names, stateless)

        logger.info(f"Starting server for agents on {host}:{port}")
        logger.info(f"- {len(cls.list_agents())} registered agents: {', '.join(cls.list_agents())}")

        # Create uvicorn server configuration
        config = uvicorn.Config(
            cls._app,
            host=host,
            port=port,
            log_level=log_level,
            loop="asyncio"
        )

        # Create and run the server
        server = uvicorn.Server(config)
        await server.serve()

    @classmethod
    def serve_in_thread(cls,
                        agents: Union[Agent, List[Agent]],
                        host: str = "0.0.0.0",
                        port: int = 1333,
                        stateless: bool = True,
                        model_names: Optional[Union[str, List[str]]] = None,
                        log_level: str = "info") -> None:
        """
        Serve SDialog agents in a separate thread (alternative for Jupyter).

        This method runs the server in a separate thread, allowing it to coexist
        with Jupyter's event loop without conflicts. It's automatically used as
        a fallback by the main serve() method when an event loop conflict is detected.

        :param agents: The SDialog agent or a list of agents to serve.
        :type agents: Union[Agent, List[Agent]]
        :param host: Host address to bind the server to.
        :type host: str
        :param port: Port number to bind the server to.
        :type port: int
        :param stateless: If True, the agent will not maintain memory between requests and the
                          full context must be provided with each request.
        :type stateless: bool
        :param model_names: Model names to expose in the API (defaults to agent's name).
        :type model_names: Optional[Union[str, List[str]]]
        :param log_level: Logging level for the server.
        :type log_level: str
        :return: The thread object running the server.
        :rtype: threading.Thread
        """
        import threading

        def run_server():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    cls.serve_async(agents, host, port, stateless, model_names, log_level)
                )
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
            finally:
                loop.close()

        logger.info("Starting threaded server...")
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

        logger.info("Server thread started. Use Ctrl+C in the main process to stop.")
        return thread

    @classmethod
    def _create_app(cls) -> None:
        """Create and configure the FastAPI application."""
        cls._app = FastAPI(
            title="SDialog OpenAI-Compatible API",
            description="OpenAI-compatible API for SDialog agents",
            version="1.0.0"
        )

        # Add CORS middleware
        cls._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        cls._register_routes()

    @classmethod
    def _register_routes(cls) -> None:
        """Register API routes."""

        @cls._app.get("/v1/models")
        async def list_models():
            """List available models."""
            models = []
            for model_name in cls._agents.keys():
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "sdialog"
                })
            return {"object": "list", "data": models}

        @cls._app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle chat completion requests."""
            return await cls._non_stream_response(request)

        @cls._app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "agents": list(cls._agents.keys())}

        # Ollama-compatible endpoints
        @cls._app.get("/api/tags")
        async def ollama_list_models():
            """Ollama-compatible endpoint to list models."""
            models = []
            for model_name in cls._agents.keys():
                models.append({
                    "name": model_name,
                    "model": model_name,
                    "modified_at": datetime.now().isoformat() + "Z",
                    "size": 0,  # We don't track model size for SDialog agents
                    "digest": f"sha256:{'0' * 64}",  # Dummy digest
                    "details": {
                        "parent_model": "",
                        "format": "sdialog",
                        "family": "sdialog",
                        "families": ["sdialog"],
                        "parameter_size": "unknown",
                        "quantization_level": "none"
                    }
                })
            return {"models": models}

        @cls._app.get("/api/ps")
        async def ollama_list_running():
            """Ollama-compatible endpoint to list running models."""
            # For SDialog, all registered agents are considered "running"
            models = []
            for model_name in cls._agents.keys():
                models.append({
                    "name": model_name,
                    "model": model_name,
                    "size": 0,
                    "digest": f"sha256:{'0' * 64}",
                    "details": {
                        "parent_model": "",
                        "format": "sdialog",
                        "family": "sdialog",
                        "families": ["sdialog"],
                        "parameter_size": "unknown",
                        "quantization_level": "none"
                    },
                    "expires_at": datetime.now().isoformat() + "Z",
                    "size_vram": 0
                })
            return {"models": models}

        @cls._app.get("/api/version")
        async def ollama_version():
            """Ollama-compatible version endpoint."""
            return {"version": "0.1.0-sdialog"}

        @cls._app.post("/api/show")
        async def ollama_show_model(request: Request):
            """Ollama-compatible endpoint to show model information."""
            try:
                body = await request.json()
                model_name = body.get("name", "")

                if model_name not in cls._agents:
                    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

                agent = cls._agents[model_name]
                return {
                    "license": "MIT",
                    "modelfile": (f"# SDialog Agent: {model_name}\n"
                                  "# Persona: " + (getattr(agent.persona, 'name', 'Unknown')
                                                   if agent.persona else 'None')),
                    "parameters": {},
                    "template": "{{ .Prompt }}",
                    "details": {
                        "parent_model": "",
                        "format": "sdialog",
                        "family": "sdialog",
                        "families": ["sdialog"],
                        "parameter_size": "unknown",
                        "quantization_level": "none"
                    },
                    "model_info": {
                        "general.architecture": "sdialog",
                        "general.name": model_name
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @cls._app.post("/api/chat")
        async def ollama_chat(request: OllamaChatRequest):
            """Ollama-compatible chat endpoint."""

            logger.info(f"Ollama chat {'streaming' if request.stream else 'non-streaming'} "
                        f"request for model '{request.model}' with {len(request.messages)} messages")

            # Convert Ollama request to OpenAI format
            openai_request = ChatCompletionRequest(
                model=request.model,
                messages=request.messages,
                stream=request.stream or False,
                temperature=request.options.get("temperature") if request.options else None,
                max_tokens=request.options.get("num_predict") if request.options else None
            )

            if request.stream:
                # For streaming, we need to convert the OpenAI streaming format to Ollama format
                return StreamingResponse(
                    cls._stream_response(openai_request),
                    media_type="application/x-ndjson"
                )
            else:
                # For non-streaming, convert OpenAI response to Ollama format
                openai_response = await cls._non_stream_response(openai_request)

                if isinstance(openai_response, JSONResponse):
                    openai_data = json.loads(openai_response.body.decode())

                    ollama_response = OllamaChatResponseNonStreaming(
                        model=request.model,
                        created_at=datetime.now().isoformat() + "Z",
                        message=openai_data["choices"][0]["message"]
                    )

                    return JSONResponse(content=ollama_response.model_dump())
                else:
                    return openai_response

    @classmethod
    def _maybe_reset_agent_for_request(cls, agent: Agent, request_messages: List[ChatMessage]) -> None:
        """Heuristically detect if this request starts a new chat and reset agent memory if so.

        Strategy:
        - Compare the (user/assistant) role-content sequence from the request (excluding the last
          user message to be processed) against the suffix of the agent's current memory transcript.
        - If there's no reasonable overlap (or the client sent an empty history while the agent
          already has turns), we treat it as a new chat and call agent.reset().

        Notes:
        - System messages are ignored in the comparison because the agent's internal system prompt
          may differ from the client's.
        - Whitespace is normalized to be robust to minor formatting differences.
        """
        try:
            # Build normalized (role, content) sequence from request, keeping only user/assistant
            req_hist = [re.sub(r'\s+', ' ', m.content).strip()
                        for m in request_messages
                        if m.role in ("user", "assistant") and m.content.strip()]
            req_hist = req_hist[:-1] if request_messages and request_messages[-1].role == "user" else req_hist

            # Build normalized (role, content) sequence from agent memory, ignoring system/tool messages
            mem_hist = []
            for msg in agent.memory:
                # LangChain message classes expose type via class name
                cls_name = msg.__class__.__name__
                if cls_name != "HumanMessage" and cls_name != "AIMessage":
                    continue
                if msg.content.strip():
                    mem_hist.append(re.sub(r'\s+', ' ', msg.content).strip())

            if len(mem_hist) != len(req_hist) or mem_hist != req_hist:
                logger.info("[New Chat Session Detected]")
                agent.reset()
        except Exception:
            # Fail-safe: never crash on best-effort detection
            return

    @classmethod
    def _get_input_from_request(cls, agent: Agent, request: ChatCompletionRequest) -> Optional[Union[str, List]]:
        if cls._stateless:
            return [_role2msg_class[msg.role](content=msg.content)
                    for msg in request.messages
                    if msg.role in _role2msg_class]
        else:
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            last_user_message = user_messages[-1].content if user_messages else ""

            # If Open WebUI task message, ignore
            if last_user_message.startswith("### Task:"):
                logger.info("Open WebUI task message detected [ignored]")
                return None

            cls._maybe_reset_agent_for_request(agent, request.messages)

            return last_user_message

    @classmethod
    async def _non_stream_response(cls, request: ChatCompletionRequest) -> Union[JSONResponse, StreamingResponse]:
        """
        Handle chat completion requests.

        :param request: The chat completion request.
        :type request: ChatCompletionRequest
        :return: The completion response.
        :rtype: Union[JSONResponse, StreamingResponse]
        """
        # Check if model exists
        if request.model not in cls._agents:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not found. Available models: {list(cls._agents.keys())}"
            )

        agent = cls._agents[request.model]
        agent_lock = cls._agent_locks[request.model]

        try:
            with agent_lock:
                user_input = cls._get_input_from_request(agent, request)

                # If Open WebUI task message, ignore
                if user_input is None:
                    logger.info("Open WebUI task message detected [ignored]")
                    events = []
                else:
                    # Generate response with events
                    events = agent(user_input, return_events=True)

                return cls._create_response(request, events)

        except Exception as e:
            if type(e) is HTTPException and e.status_code == 400:
                raise e
            else:
                logger.error(f"Error processing chat completion: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @classmethod
    def _create_response(cls, request: ChatCompletionRequest, events: List) -> JSONResponse:
        """
        Create a non-streaming chat completion response with separate messages for reasoning, tools, and output.

        :param request: The original request.
        :type request: ChatCompletionRequest
        :param events: List of events from the agent.
        :type events: List
        :return: The completion response.
        :rtype: JSONResponse
        """
        # TODO: Implement as with _stream_response.
        #       For now, return just the generated text, no events
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(datetime.now().timestamp())

        content = "" if not events else events[-1].content if events[-1].action == "utter" else ""

        # If no choices were created, create a default empty response
        default_message = ChatMessage(role="assistant", content=content)
        choices = [ChatCompletionChoice(
            index=0,
            message=default_message,
            finish_reason="stop"
        )]

        response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=choices
        )

        return JSONResponse(content=response.model_dump())

    @classmethod
    async def _stream_response(cls, request: ChatCompletionRequest):
        """
        Create an Ollama-compatible streaming chat response.

        :param request: The chat completion request.
        :type request: ChatCompletionRequest
        :return: Generator yielding Ollama streaming response chunks.
        """
        # Check if model exists
        if request.model not in cls._agents:
            error_response = {
                "error": f"Model '{request.model}' not found. Available models: {list(cls._agents.keys())}"
            }
            yield json.dumps(error_response) + "\n"
            return

        agent = cls._agents[request.model]
        agent_lock = cls._agent_locks[request.model]

        try:
            with agent_lock:
                user_input = cls._get_input_from_request(agent, request)

                # If Open WebUI task message, ignore
                if user_input is None:
                    logger.info("Open WebUI task message detected [ignored]")
                    events = []
                else:
                    # Generate response with events
                    events = agent(user_input, return_events=True)

                # Pre-index tool outputs by call_id for quick lookup when we see the corresponding call event
                outputs_by_call_id = {}
                for e in events:
                    if hasattr(e, 'action') and e.action == "tool" and getattr(e, 'actionLabel', None) == "output":
                        try:
                            content = e.content if isinstance(e.content, dict) else {}
                            call_id = content.get("call_id")
                            if call_id is not None:
                                outputs_by_call_id[call_id] = content.get("output")
                        except Exception:
                            # Be resilient to any malformed event content
                            pass

                def _to_markdown_code_block(value: Any) -> str:
                    """Pretty print value as a Markdown fenced code block.
                    Uses json formatting if possible; falls back to text.
                    """
                    # Try to detect dict/list directly
                    try:
                        if isinstance(value, (dict, list)):
                            return f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```"
                        # Try to parse strings as JSON
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                                return f"```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
                            except Exception:
                                # Not JSON, return as plain text block
                                return f"```\n{value}\n```"
                        # Fallback for other types
                        return f"```\n{str(value)}\n```"
                    except Exception:
                        return f"```\n{str(value)}\n```"

                def _to_call_signature(name: str, args: Any) -> str:
                    """Format a function-style call: name(k1=v1, k2=v2, ...)."""
                    try:
                        if isinstance(args, dict):
                            parts = []
                            for k, v in args.items():
                                if isinstance(v, (dict, list)):
                                    v_str = json.dumps(v, ensure_ascii=False)
                                elif isinstance(v, str):
                                    v_str = '"' + v.replace('"', '\\"') + '"'
                                else:
                                    v_str = str(v)
                                parts.append(f"{k}={v_str}")
                            call = f"{name}(" + ", ".join(parts) + ")"
                        else:
                            # Non-dict args: best effort serialize inside single positional arg
                            if not isinstance(args, str):
                                v_str = json.dumps(args, ensure_ascii=False)
                            else:
                                v_str = '"' + args.replace('"', '\\"') + '"'
                            call = f"{name}({v_str})"
                    except Exception:
                        call = f"{name}()"

                    return f"```python\n{call}\n```"

                def _send_chunk(content: str = None, thinking: str = None,
                                tool_calls: List[Dict] = None, tool_call_id: str = None,
                                done: bool = False):
                    """Helper to create and yield a response chunk."""
                    chunk = {
                        "model": request.model,
                        "created_at": datetime.now().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                        },
                        "done": done
                    }
                    if content is not None:
                        chunk["message"]["content"] = content
                    if thinking is not None:
                        chunk["message"]["thinking"] = thinking
                    if tool_calls is not None:
                        chunk["message"]["tool_calls"] = tool_calls
                    if tool_call_id is not None:
                        chunk["message"]["role"] = "tool"  # Not supported by Open WebUI
                        chunk["message"]["tool_call_id"] = tool_call_id
                    return json.dumps(chunk) + "\n"

                # Stream events in Ollama format with structured reasoning and tool support
                for event in events:
                    content = ""

                    if event.action == "think":
                        # Reasoning/thinking events with structured format
                        # yield _send_chunk(thinking=event.content)
                        content = f"<think>\n{event.content}\n</think>"
                    elif event.action.startswith("instruct"):
                        # Orchestration instructions (one-time or persistent)
                        scope = "persistent" if "persist" in event.action else "one-time"
                        content = (f"<think>\n* **Orchestration ({event.actionLabel}):** _({scope} instruction)_\n"
                                   f"{event.content}\n</think>")
                        # yield _send_chunk(thinking=content)
                    elif event.action == "tool" and event.actionLabel == "call":
                        # Tool events with structured format
                        call_info = event.content if isinstance(event.content, dict) else {}
                        tool_name = call_info.get('name', 'unknown')
                        tool_args = call_info.get('args', {})
                        call_id = call_info.get('id')
                        # Find matching output for this call id, if any
                        out_value = outputs_by_call_id.get(call_id)
                        if out_value is not None:
                            out_block = _to_markdown_code_block(out_value)
                        else:
                            out_block = "_(no output)_"
                        content = (f"<think>\n{_to_call_signature(tool_name, tool_args)}\n"
                                   f"* **Output:**\n{out_block}\n</think>")
                        # # Tool call message (role=assistant) with call info
                        # _send_chunk(tool_calls=[{
                        #     "id": call_id,
                        #     "index": tool_call_ix,
                        #     "function": {
                        #         "name": tool_name,
                        #         "arguments": tool_args
                        #     },
                        # }])
                        # tool_call_ix += 1
                        # # Tool result message (role=tool) with output
                        # _send_chunk(content=out_value, tool_call_id=call_id)
                    elif event.action == "utter":
                        # Main utterance - stream word by word
                        content = event.content

                    if content:
                        # Stream while preserving original whitespace separators (spaces, newlines, tabs)
                        for match in re.finditer(r'\S+\s*', content):
                            yield _send_chunk(content=match.group(0))
                            time.sleep(0.01)  # Adjust delay as needed

                # Send final chunk
                yield _send_chunk(content="", done=True)

        except Exception as e:
            error_response = {"error": f"Internal server error: {str(e)}"}
            yield json.dumps(error_response) + "\n"

    @classmethod
    def add_agent(cls, agent: Agent, model_name: str = None) -> None:
        """
        Add an agent to the server without starting it.

        :param agent: The SDialog agent to add.
        :type agent: Agent
        :param model_name: Model name to use for the agent.
        :type model_name: str
        """
        model_name = cls._setup_agents(agent, model_name)
        logger.info(f"Added agent '{model_name}' to server")

    @classmethod
    def remove_agent(cls, model_name: str) -> None:
        """
        Remove an agent from the server.

        :param model_name: Model name of the agent to remove.
        :type model_name: str
        """
        if model_name in cls._agents:
            del cls._agents[model_name]
            del cls._agent_locks[model_name]
            logger.info(f"Removed agent '{model_name}' from server")

    @classmethod
    def list_agents(cls) -> List[str]:
        """
        List all registered agent model names.

        :return: List of model names.
        :rtype: List[str]
        """
        return list(cls._agents.keys())

    @classmethod
    def reset_agent(cls, model_name: str, seed: Optional[int] = None) -> None:
        """
        Reset an agent's memory and state.

        :param model_name: Model name of the agent to reset.
        :type model_name: str
        :param seed: Optional seed for the reset.
        :type seed: Optional[int]
        """
        if model_name in cls._agents:
            with cls._agent_locks[model_name]:
                cls._agents[model_name].reset(seed=seed)
            logger.info(f"Reset agent '{model_name}'")
        else:
            raise ValueError(f"Agent '{model_name}' not found")
