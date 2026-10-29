# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
"""
FastAPI webhook server for SDialog WhatsApp agent serving.
"""

import hmac
import hashlib
import asyncio
import logging
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

from .config import WhatsAppConfig, AUTO_SESSION_TIMEOUT, AUTO_SESSION_TIMEOUT_NEW_CHAT
from .client import MetaWhatsAppClient
from .session_store import SQLiteSessionStore
from ..rate_limit import RateLimiter

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL = 86400  # Seconds between purges of old idempotency records and inactive sessions


class WhatsAppServer:
    """
    FastAPI server exposing Meta WhatsApp Cloud API webhook endpoints for SDialog agents.
    """

    _app: Optional[FastAPI] = None
    _config: Optional[WhatsAppConfig] = None
    _agent: Any = None
    _store: Optional[SQLiteSessionStore] = None
    _client: Optional[MetaWhatsAppClient] = None
    _agent_lock: Optional[asyncio.Lock] = None
    _rate_limiter: Optional[RateLimiter] = None  # Per sender ID
    _rate_notified: Dict[Tuple[str, str], float] = {}  # (sender ID, "minute"/"hour") -> time of last notice
    _new_chat_events: bool = False  # Whether Meta signals new chats (welcome messages enabled on the number)
    _last_cleanup: float = 0.0

    @classmethod
    def get_agent_lock(cls) -> asyncio.Lock:
        """
        Get the lock guarding the shared agent.

        A single agent instance serves all users, swapping each user's state in and out of its memory,
        so turns must be processed one at a time. The lock is FIFO, so messages from the same user are
        still processed in arrival order. Created lazily so it binds to the server's event loop.
        """
        if cls._agent_lock is None:
            cls._agent_lock = asyncio.Lock()
        return cls._agent_lock

    @classmethod
    def create_app(cls, agent: Any, config: WhatsAppConfig) -> FastAPI:
        """Create and configure the FastAPI WhatsApp webhook application."""
        config.validate(dry_run=True)
        if not config.enforce_signature:
            logger.warning("[WhatsApp] enforce_signature=False: webhook requests are NOT verified, anyone who "
                           "knows the URL can make the agent reply (and spend LLM/WhatsApp credits) on your behalf")

        cls._agent = agent
        cls._config = config
        cls._store = SQLiteSessionStore(db_path=config.db_path)
        cls._client = MetaWhatsAppClient(config)
        cls._agent_lock = None
        cls._rate_limiter = RateLimiter({"hour": (config.max_messages_per_hour, 3600),
                                         "minute": (config.max_messages_per_minute, 60)})
        cls._rate_notified.clear()
        cls._new_chat_events = False
        cls._last_cleanup = 0.0
        cls._cleanup_if_due()

        app = FastAPI(
            title="SDialog WhatsApp Agent Server",
            description="Meta WhatsApp Cloud API Webhook Server for SDialog Agents",
            version="1.0.0",
        )

        @app.get(config.webhook_path)
        async def verify_webhook(
            hub_mode: str = Query(None, alias="hub.mode"),
            hub_verify_token: str = Query(None, alias="hub.verify_token"),
            hub_challenge: str = Query(None, alias="hub.challenge"),
        ):
            """Handle Meta WhatsApp webhook verification challenge."""
            if hub_mode == "subscribe" and hub_verify_token and \
               hmac.compare_digest(hub_verify_token, config.verify_token):
                logger.info("[WhatsApp] Webhook verification challenge accepted")
                return Response(content=hub_challenge, media_type="text/plain")
            logger.warning("[WhatsApp] Webhook verification failed: token mismatch")
            return Response(content="Verification failed", status_code=403)

        @app.post(config.webhook_path)
        async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
            """
            Receive incoming WhatsApp webhook events.

            Validates HMAC signature, checks idempotency, and queues background processing.
            Returns 200 OK immediately to satisfy Meta's 3-second SLA.
            """
            raw_body = await request.body()

            # 1. HMAC Signature Verification
            if config.enforce_signature:
                sig_header = request.headers.get("X-Hub-Signature-256", "")
                if not cls._verify_signature(raw_body, sig_header, config.app_secret):
                    logger.warning("[WhatsApp] Rejected request with invalid HMAC signature")
                    raise HTTPException(status_code=401, detail="Invalid signature")

            cls._cleanup_if_due()

            # 2. Parse payload
            try:
                payload = await request.json()
            except Exception:
                return JSONResponse({"status": "invalid_json"}, status_code=400)

            # 3. Guard: Extract messages from Meta payload structure
            messages, sender_name = cls._extract_messages(payload)
            if not messages:
                # Normal delivery status update (sent, delivered, read) - acknowledge and ignore
                return {"status": "ok"}

            for msg in messages:
                wamid = msg.get("id")
                sender_id = msg.get("from")
                msg_type = msg.get("type")

                if not sender_id or not wamid:
                    continue

                # 4. Idempotency check: drop duplicate webhooks from Meta retries
                if cls._store.is_message_processed(wamid):
                    logger.info(f"[WhatsApp] Duplicate message {wamid} ignored")
                    continue

                cls._store.mark_message_processed(wamid)

                # 5. Per-user rate limit, so one user can't flood the (serialized) agent or run up costs
                allowed, notice = cls._check_rate_limit(sender_id)
                if not allowed:
                    logger.warning(f"[WhatsApp] Rate limit exceeded for {sender_id}, message {wamid} dropped")
                    if notice:
                        background_tasks.add_task(cls._client.send_text_message, to=sender_id, text=notice)
                    continue

                # 6. Route by message type
                if msg_type == "text":
                    text_body = msg.get("text", {}).get("body", "").strip()
                    if text_body:
                        background_tasks.add_task(
                            cls._handle_user_turn,
                            sender_id=sender_id,
                            text=text_body,
                            wamid=wamid,
                        )
                elif msg_type == "request_welcome":
                    # The user opened a new chat (only sent if welcome messages are enabled on the number)
                    background_tasks.add_task(cls._handle_new_chat, sender_id=sender_id)
                else:
                    # Non-text media (audio, image, document, location, etc.)
                    background_tasks.add_task(
                        cls._handle_unsupported_media,
                        sender_id=sender_id,
                        msg_type=msg_type,
                    )

            return {"status": "ok"}

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "service": "sdialog-whatsapp"}

        cls._app = app
        return app

    @classmethod
    async def _handle_user_turn(cls, sender_id: str, text: str, wamid: str) -> None:
        """Process incoming user utterance under the agent lock and reply via WhatsApp."""
        # Mark message as read (blue checkmark) if configured
        if cls._config.mark_as_read and cls._client:
            await cls._client.mark_as_read(wamid)

        async with cls.get_agent_lock():
            try:
                # Check for explicit session reset commands
                if text.lower() in ("/reset", "/clear", "/restart"):
                    cls._store.reset_session(sender_id)
                    await cls._client.send_text_message(
                        to=sender_id,
                        text="Your conversation has been reset. How can I help you?",
                    )
                    return

                # Load user state from SQLite
                state = cls._store.load_session(sender_id, timeout_seconds=cls._session_timeout())

                # A finished conversation (agent said goodbye) starts over on the next message
                if state and state.get("finished"):
                    state = None

                # Reset agent memory and restore user session state
                cls._agent.reset()
                if state:
                    cls._agent.set_state(state)

                # Run inference in a threadpool so synchronous LLMs don't block the event loop
                logger.debug(f"[WhatsApp] Processing turn for user {sender_id}: '{text[:30]}...'")
                response = await asyncio.to_thread(cls._agent, text)

                if not response:
                    logger.warning(f"[WhatsApp] Agent returned empty response for {sender_id}")
                    return

                # Save updated user state to SQLite
                new_state = cls._agent.get_state()
                new_state["memory"] = cls._trim_memory(new_state["memory"], cls._config.max_history_turns)
                cls._store.save_session(sender_id, new_state)

                # Dispatch reply via Meta WhatsApp Cloud API
                await cls._client.send_text_message(to=sender_id, text=response)

            except Exception as e:
                logger.error(f"[WhatsApp] Error processing message for {sender_id}: {e}", exc_info=True)
                try:
                    await cls._client.send_text_message(
                        to=sender_id,
                        text="I encountered a problem processing your request. Please try again in a moment.",
                    )
                except Exception:
                    pass

    @classmethod
    async def _handle_new_chat(cls, sender_id: str) -> None:
        """Start a fresh conversation when the user opens a new chat, and greet them."""
        if not cls._new_chat_events:
            cls._new_chat_events = True
            if cls._config.session_timeout == "auto":
                logger.info("[WhatsApp] Meta signals new chats, so conversations now start over when the user "
                            f"opens a new chat (or after {AUTO_SESSION_TIMEOUT_NEW_CHAT}s of inactivity)")

        async with cls.get_agent_lock():
            try:
                cls._store.reset_session(sender_id)
                cls._agent.reset()
                # With no user message, the agent replies with its first_utterance if set, else the LLM greets
                greeting = await asyncio.to_thread(cls._agent)
                if not greeting:
                    return
                cls._store.save_session(sender_id, cls._agent.get_state())
                await cls._client.send_text_message(to=sender_id, text=greeting)
            except Exception as e:
                logger.error(f"[WhatsApp] Error greeting {sender_id} on a new chat: {e}", exc_info=True)

    @classmethod
    async def _handle_unsupported_media(cls, sender_id: str, msg_type: str) -> None:
        """Notify user that non-text media messages are currently unsupported."""
        logger.info(f"[WhatsApp] Received unsupported message type '{msg_type}' from {sender_id}")
        if cls._client:
            await cls._client.send_text_message(
                to=sender_id,
                text="I currently only support text messages. Please send your query as text.",
            )

    @classmethod
    def _session_timeout(cls) -> Optional[int]:
        """Idle seconds before a conversation starts over (None: never), resolving session_timeout=\"auto\"."""
        if cls._config.session_timeout == "auto":
            return AUTO_SESSION_TIMEOUT_NEW_CHAT if cls._new_chat_events else AUTO_SESSION_TIMEOUT
        return cls._config.session_timeout

    @classmethod
    def _cleanup_if_due(cls) -> None:
        """Purge old idempotency records and inactive sessions, at most once every CLEANUP_INTERVAL seconds."""
        now = time.monotonic()
        if cls._last_cleanup and now - cls._last_cleanup < CLEANUP_INTERVAL:
            return
        cls._last_cleanup = now
        try:
            cls._store.cleanup_old_records(session_retention_days=cls._config.session_retention_days)
        except Exception as e:
            logger.warning(f"[WhatsApp] Cleanup of old records failed: {e}")

    @classmethod
    def _check_rate_limit(cls, sender_id: str) -> Tuple[bool, Optional[str]]:
        """
        Per-sender sliding window rate limit: a per-minute limit for bursts and a per-hour limit for
        sustained use. Rejected messages do not count towards the limits.

        :return: ``(allowed, notice)``, where ``notice`` is the text to send back for the first rejected message
                 in each window (None otherwise), so the user gets a single notice instead of one per message.
        """
        exceeded = cls._rate_limiter.hit(sender_id)
        if exceeded is None:
            return True, None

        unit, _ = exceeded
        now = time.monotonic()
        last_notice = cls._rate_notified.get((sender_id, unit))
        if last_notice is not None and now - last_notice < (60 if unit == "minute" else 3600):
            return False, None
        cls._rate_notified[(sender_id, unit)] = now
        if unit == "minute":
            return False, "You are sending messages too quickly. Please wait a minute and try again."
        return False, "You have reached the message limit for now. Please try again later."

    @staticmethod
    def _trim_memory(memory: List[Dict[str, Any]], max_messages: int) -> List[Dict[str, Any]]:
        """
        Keep at most the last ``max_messages`` serialized messages, starting at a user message
        so that tool calls are never separated from their tool results.
        """
        if len(memory) <= max_messages:
            return memory
        memory = memory[-max_messages:]
        while memory and memory[0].get("type") != "human":
            memory = memory[1:]
        return memory

    @staticmethod
    def _verify_signature(raw_body: bytes, sig_header: str, app_secret: str) -> bool:
        """Verify the X-Hub-Signature-256 header using HMAC-SHA256 in constant time."""
        if not sig_header or not sig_header.startswith("sha256="):
            return False
        expected_hash = sig_header[7:]
        computed_hash = hmac.new(
            app_secret.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed_hash, expected_hash)

    @staticmethod
    def _extract_messages(payload: Dict[str, Any]):
        """
        Safely extract message objects and contact details from Meta webhook payload.
        Meta may batch several entries and changes in a single delivery, so all of them are collected.
        """
        messages = []
        sender_name = None
        try:
            for entry in payload.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages.extend(value.get("messages", []))
                    contacts = value.get("contacts", [])
                    if contacts and sender_name is None:
                        sender_name = contacts[0].get("profile", {}).get("name")
        except (KeyError, AttributeError, TypeError):
            pass
        return messages, sender_name

    @classmethod
    def serve(cls, agent: Any, config: Optional[WhatsAppConfig] = None) -> None:
        """Start the WhatsApp webhook server with optional auto-discovery and webhook sync."""
        config = config or WhatsAppConfig()

        # 1. Auto-discover Phone Number ID if missing but access token is present
        if not config.phone_number_id and config.access_token:
            from .discovery import MetaDiscoveryClient
            discovery = MetaDiscoveryClient(config.access_token, config.app_secret)
            try:
                config.phone_number_id = asyncio.run(discovery.discover_phone_number_id())
            except Exception as e:
                logger.debug(f"[WhatsApp] Auto-discovery skipped or failed: {e}")

        config.validate(dry_run=False)

        # 2. Optional tunnel setup for local testing
        public_url = None
        if config.tunnel:
            public_url = cls._start_tunnel(config)

        # 3. Auto-configure Meta webhooks if requested and tunnel/public URL is active
        if config.auto_setup and public_url and config.app_secret:
            from .discovery import MetaDiscoveryClient
            discovery = MetaDiscoveryClient(config.access_token, config.app_secret)
            callback_url = f"{public_url}{config.webhook_path}"
            logger.warning(f"[WhatsApp] auto_setup=True: overriding the Meta webhook URL of the WhatsApp Business "
                           f"Account and phone number to {callback_url} (it is not restored on shutdown)")
            try:
                res = asyncio.run(discovery.configure_webhooks(
                    callback_url=callback_url,
                    verify_token=config.verify_token,
                    phone_number_id=config.phone_number_id
                ))
                logger.info(f"[WhatsApp] Auto-configuration results: {res}")
            except Exception as e:
                logger.warning(f"[WhatsApp] Auto-configuration encountered an error: {e}")

        app = cls.create_app(agent, config)

        # 4. Check whether Meta signals new chats, so "auto" sessions can start over on them
        if config.session_timeout == "auto":
            from .discovery import MetaDiscoveryClient
            try:
                enabled = asyncio.run(MetaDiscoveryClient(config.access_token).is_welcome_message_enabled(
                    config.phone_number_id))
            except Exception as e:
                logger.debug(f"[WhatsApp] Welcome message detection failed: {e}")
                enabled = None
            if enabled:
                cls._new_chat_events = True
                logger.info("[WhatsApp] Welcome messages are enabled: conversations start over when the user opens "
                            f"a new chat (or after {AUTO_SESSION_TIMEOUT_NEW_CHAT}s of inactivity)")
            else:
                logger.info(f"[WhatsApp] Conversations start over after {AUTO_SESSION_TIMEOUT}s of inactivity "
                            "(enable welcome messages on the phone number to start over on new chats instead)")

        logger.info(f"[WhatsApp] Starting server on {config.host}:{config.port}")
        logger.info(f"[WhatsApp] Webhook endpoint: {config.webhook_path}")
        if config.verify_token_generated:
            logger.warning(f"[WhatsApp] No verify token set, generated one for this run: {config.verify_token} "
                           "(enter it in Meta's webhook settings, or set WHATSAPP_VERIFY_TOKEN to keep it fixed)")

        uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)

    @classmethod
    def serve_in_thread(cls, agent: Any, config: Optional[WhatsAppConfig] = None) -> threading.Thread:
        """Start the WhatsApp webhook server in a background daemon thread (useful for Jupyter / tests)."""
        config = config or WhatsAppConfig()
        app = cls.create_app(agent, config)

        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            uv_config = uvicorn.Config(
                app,
                host=config.host,
                port=config.port,
                log_level=config.log_level,
                loop="asyncio",
            )
            server = uvicorn.Server(uv_config)
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info(f"[WhatsApp] Threaded server started on {config.host}:{config.port}")
        return thread

    @staticmethod
    def _start_tunnel(config: WhatsAppConfig) -> Optional[str]:
        """Attempt to spawn an ngrok tunnel if pyngrok is installed. Returns public URL."""
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(config.port, bind_tls=True).public_url
            logger.info("=" * 60)
            logger.info(f"[WhatsApp] Public Tunnel URL: {public_url}{config.webhook_path}")
            logger.info(f"[WhatsApp] Meta Callback URL: {public_url}{config.webhook_path}")
            logger.info("=" * 60)
            return public_url
        except ImportError:
            logger.warning(
                "[WhatsApp] tunnel=True requested, but 'pyngrok' is not installed. "
                "Install it with `pip install pyngrok` or use a tool like ngrok/cloudflared manually."
            )
            return None
