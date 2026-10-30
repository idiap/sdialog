# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
"""
Unit and integration tests for SDialog WhatsApp server.
"""

import hmac
import hashlib
import json
import os
import tempfile
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, messages_to_dict, messages_from_dict

from sdialog.whatsapp import (
    WhatsAppConfig,
    WhatsAppServer,
    SQLiteSessionStore,
    MetaWhatsAppClient,
)


class MockWhatsAppAgent:
    """Mock agent for WhatsApp testing (mirrors Agent's in-place memory reset and state API)."""

    def __init__(self, name="TestBot", latency=0.0):
        self.name = name
        self.latency = latency
        self.memory = [SystemMessage(content="You are a helpful test bot.")]
        self.call_count = 0
        self.last_input = None
        self._finished = False

    def __call__(self, message: str = "", **kwargs) -> str:
        self.call_count += 1
        self.last_input = message
        if message:
            self.memory.append(HumanMessage(content=message))
        time.sleep(self.latency)
        seen = [m.content for m in self.memory if isinstance(m, HumanMessage)]
        reply = f"Bot reply to: {' | '.join(seen)}" if seen else "Bot greeting"
        self.memory.append(AIMessage(content=reply))
        return reply

    def reset(self, seed=None):
        self.memory[:] = self.memory[:1]
        self._finished = False

    def clone(self):
        return MockWhatsAppAgent(name=self.name, latency=self.latency)

    def get_state(self):
        return {"version": 1, "memory": messages_to_dict(self.memory[1:]),
                "finished": self._finished, "orchestrators": []}

    def set_state(self, state):
        self.memory[:] = self.memory[:1] + messages_from_dict(state["memory"])
        self._finished = state["finished"]


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_config(temp_db):
    """WhatsAppConfig configured for testing."""
    return WhatsAppConfig(
        port=1234,
        phone_number_id="1234567890",
        access_token="test_access_token",
        verify_token="test_verify_token",
        app_secret="test_app_secret",
        db_path=temp_db,
        session_timeout=3600,
        enforce_signature=True,
        mark_as_read=False,  # avoid real Graph API calls from background tasks
        max_workers=1,  # serve with the agent itself, so tests can inspect it
    )


# ---------------------------------------------------------------------------
# Test 1: Config
# ---------------------------------------------------------------------------

def test_whatsapp_config_defaults():
    config = WhatsAppConfig(verify_token="custom_token")
    assert config.port == 1234
    assert config.verify_token == "custom_token"
    assert config.session_timeout == "auto"
    assert config.session_retention_days == 30
    assert config.max_history_turns == 40


@pytest.fixture
def no_whatsapp_env(monkeypatch):
    for var in ("WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_ACCESS_TOKEN", "WHATSAPP_TOKEN", "WHATSAPP_VERIFY_TOKEN",
                "WHATSAPP_WEBHOOK_VERIFY_TOKEN", "WHATSAPP_APP_SECRET"):
        monkeypatch.delenv(var, raising=False)


def test_whatsapp_config_validation(no_whatsapp_env):
    config = WhatsAppConfig(phone_number_id=None, access_token=None, app_secret="secret")
    with pytest.raises(ValueError, match="phone_number_id is required"):
        config.validate(dry_run=False)

    config = WhatsAppConfig(phone_number_id="123", access_token=None, app_secret="secret")
    with pytest.raises(ValueError, match="access_token is required"):
        config.validate(dry_run=False)

    # Valid config
    config = WhatsAppConfig(phone_number_id="123", access_token="token", app_secret="secret")
    config.validate(dry_run=False)  # Should not raise


def test_whatsapp_config_requires_app_secret(no_whatsapp_env, temp_db):
    config = WhatsAppConfig(phone_number_id="123", access_token="token", db_path=temp_db)
    with pytest.raises(ValueError, match="app_secret is required"):
        config.validate(dry_run=False)
    with pytest.raises(ValueError, match="app_secret is required"):
        WhatsAppServer.create_app(MockWhatsAppAgent(), config)

    # Explicit opt-out accepts unsigned requests
    config = WhatsAppConfig(phone_number_id="123", access_token="token", db_path=temp_db,
                            enforce_signature=False, mark_as_read=False)
    client = TestClient(WhatsAppServer.create_app(MockWhatsAppAgent(), config))
    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        resp = client.post("/webhook", json=_create_meta_payload())
    assert resp.status_code == 200


def test_whatsapp_config_env_fallback_with_explicit_none(no_whatsapp_env, monkeypatch):
    """Agent.serve_whatsapp() passes None for unset arguments; env vars must still be used."""
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "env_phone")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "env_token")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "env_secret")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "env_verify")

    config = WhatsAppConfig(phone_number_id=None, access_token=None, app_secret=None, verify_token=None)
    assert (config.phone_number_id, config.access_token, config.app_secret, config.verify_token) == \
        ("env_phone", "env_token", "env_secret", "env_verify")
    assert not config.verify_token_generated


def test_whatsapp_config_generates_random_verify_token(no_whatsapp_env):
    config1, config2 = WhatsAppConfig(), WhatsAppConfig()
    assert config1.verify_token_generated
    assert len(config1.verify_token) >= 32
    assert config1.verify_token != config2.verify_token
    assert not WhatsAppConfig().auto_setup


# ---------------------------------------------------------------------------
# Test 2: SQLiteSessionStore
# ---------------------------------------------------------------------------

def test_session_store_save_load(temp_db):
    store = SQLiteSessionStore(db_path=temp_db)
    wa_id = "15551234567"

    # Initially empty
    assert store.load_session(wa_id) is None

    # Save and reload
    state = {"version": 1, "memory": [{"type": "HumanMessage", "content": "Hello"}]}
    store.save_session(wa_id, state)

    loaded = store.load_session(wa_id)
    assert loaded is not None
    assert loaded["memory"][0]["content"] == "Hello"

    # Reset
    store.reset_session(wa_id)
    assert store.load_session(wa_id) is None


def test_session_store_timeout_expiration(temp_db):
    store = SQLiteSessionStore(db_path=temp_db)
    wa_id = "15551234567"

    state = {"version": 1, "memory": []}
    store.save_session(wa_id, state)

    # With timeout_seconds=0, it should be expired immediately
    time.sleep(0.01)
    expired = store.load_session(wa_id, timeout_seconds=0)
    assert expired is None


def test_session_store_idempotency(temp_db):
    store = SQLiteSessionStore(db_path=temp_db)
    wamid = "wamid.HBgLMTIzNDU2"

    assert not store.is_message_processed(wamid)
    store.mark_message_processed(wamid)
    assert store.is_message_processed(wamid)

    # Idempotent re-marking
    store.mark_message_processed(wamid)
    assert store.is_message_processed(wamid)


def test_session_store_cleanup_old_records(temp_db):
    import sqlite3

    store = SQLiteSessionStore(db_path=temp_db)
    store.mark_message_processed("wamid.new")
    conn = sqlite3.connect(temp_db)
    with conn:
        conn.execute("INSERT INTO processed_messages (wamid, received_at) VALUES ('wamid.old', '2000-01-01 00:00:00')")
    conn.close()

    store.cleanup_old_records(retention_days=7)
    assert store.is_message_processed("wamid.new")
    assert not store.is_message_processed("wamid.old")


def test_extract_messages_from_batched_payload():
    payload = _create_meta_payload(sender="1111", wamid="wamid.A")
    payload["entry"].append(_create_meta_payload(sender="2222", wamid="wamid.B")["entry"][0])

    messages, sender_name = WhatsAppServer._extract_messages(payload)
    assert [m["id"] for m in messages] == ["wamid.A", "wamid.B"]
    assert sender_name == "Alice"
    assert WhatsAppServer._extract_messages({"entry": "malformed"}) == ([], None)


def test_trim_memory_starts_at_user_message():
    memory = [{"type": t} for t in ["human", "ai", "human", "ai", "tool", "ai"]]

    assert WhatsAppServer._trim_memory(memory, 10) == memory
    # The last 5 would start with an AI reply, so trimming skips ahead to the next user message
    assert WhatsAppServer._trim_memory(memory, 5) == memory[2:]
    assert WhatsAppServer._trim_memory(memory, 3) == []


# ---------------------------------------------------------------------------
# Test 3: MetaWhatsAppClient
# ---------------------------------------------------------------------------

def test_client_text_chunking():
    # Verify chunking does not break words prematurely
    short_text = "Hello world"
    assert MetaWhatsAppClient._chunk_text(short_text, 100) == [short_text]

    long_text = "First paragraph.\n\nSecond paragraph."
    chunks = MetaWhatsAppClient._chunk_text(long_text, 20)
    assert len(chunks) == 2
    assert chunks[0] == "First paragraph."
    assert chunks[1] == "Second paragraph."


def test_client_send_text_mocked(test_config):
    import asyncio
    client = MetaWhatsAppClient(test_config)

    async def _test():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            success = await client.send_text_message("15551234567", "Hello from test!")
            assert success is True
            assert mock_post.called

            called_url = mock_post.call_args[0][0]
            assert "1234567890/messages" in called_url

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Test 4: Webhook Verification (GET /webhook)
# ---------------------------------------------------------------------------

def test_webhook_verification_success(test_config):
    agent = MockWhatsAppAgent()
    app = WhatsAppServer.create_app(agent, test_config)
    client = TestClient(app)

    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "test_verify_token",
            "hub.challenge": "test_challenge_code_123",
        },
    )
    assert response.status_code == 200
    assert response.text == "test_challenge_code_123"


def test_webhook_verification_failure(test_config):
    agent = MockWhatsAppAgent()
    app = WhatsAppServer.create_app(agent, test_config)
    client = TestClient(app)

    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "challenge",
        },
    )
    assert response.status_code == 403

    response = client.get("/webhook", params={"hub.mode": "subscribe", "hub.challenge": "challenge"})
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Test 5: Inbound Messages & HMAC (POST /webhook)
# ---------------------------------------------------------------------------

def _create_meta_payload(sender="15551234567", text="Hello agent", wamid="wamid.123", msg_type="text",
                         sent_seconds_ago=0):
    """Helper to create a standard Meta WhatsApp webhook payload."""
    message = {"from": sender, "id": wamid, "timestamp": str(int(time.time() - sent_seconds_ago)), "type": msg_type}
    if msg_type == "text":
        message["text"] = {"body": text}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123456",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550001",
                                "phone_number_id": "1234567890",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Alice"},
                                    "wa_id": sender,
                                }
                            ],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


def _compute_hmac(body: bytes, secret: str) -> str:
    """Helper to compute Meta X-Hub-Signature-256 header value."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_webhook_hmac_enforcement(test_config):
    agent = MockWhatsAppAgent()
    app = WhatsAppServer.create_app(agent, test_config)
    client = TestClient(app)

    payload = json.dumps(_create_meta_payload()).encode()

    # 1. Missing signature
    resp = client.post("/webhook", content=payload, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401

    # 2. Invalid signature
    resp = client.post(
        "/webhook",
        content=payload,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": "sha256=invalid"},
    )
    assert resp.status_code == 401

    # 3. Valid signature
    valid_sig = _compute_hmac(payload, test_config.app_secret)
    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        resp = client.post(
            "/webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": valid_sig},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_webhook_status_receipt_ignored(test_config):
    agent = MockWhatsAppAgent()
    app = WhatsAppServer.create_app(agent, test_config)
    client = TestClient(app)

    # Status receipt payload (e.g. message delivered)
    status_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123456",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "statuses": [{"id": "wamid.123", "status": "delivered"}],
                        },
                    }
                ],
            }
        ],
    }
    raw = json.dumps(status_payload).encode()
    sig = _compute_hmac(raw, test_config.app_secret)

    resp = client.post(
        "/webhook",
        content=raw,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert agent.call_count == 0


def test_webhook_idempotency_duplicate_dropped(test_config):
    agent = MockWhatsAppAgent()
    app = WhatsAppServer.create_app(agent, test_config)
    client = TestClient(app)

    payload = json.dumps(_create_meta_payload(wamid="wamid.repeat123")).encode()
    sig = _compute_hmac(payload, test_config.app_secret)

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        # First delivery: processed
        resp1 = client.post(
            "/webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )
        assert resp1.status_code == 200

        # Second delivery (retry with same wamid): dropped
        resp2 = client.post(
            "/webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )
        assert resp2.status_code == 200
        assert agent.call_count == 1  # Agent called only once!


def test_webhook_multi_user_isolation(test_config):
    """Verify that User A and User B have separate, isolated conversation histories."""
    agent = MockWhatsAppAgent()
    app = WhatsAppServer.create_app(agent, test_config)
    client = TestClient(app)

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        # User A speaks
        pA = json.dumps(_create_meta_payload(sender="1111", text="Hello from A", wamid="wamid.A1")).encode()
        client.post(
            "/webhook",
            content=pA,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _compute_hmac(pA, test_config.app_secret),
            },
        )

        # User B speaks
        pB = json.dumps(_create_meta_payload(sender="2222", text="Hello from B", wamid="wamid.B1")).encode()
        client.post(
            "/webhook",
            content=pB,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _compute_hmac(pB, test_config.app_secret),
            },
        )

        # Check SQLite store
        store = SQLiteSessionStore(db_path=test_config.db_path)
        stateA = store.load_session("1111")
        stateB = store.load_session("2222")

        assert stateA is not None
        assert stateB is not None

        # Verify User A's history contains User A's message, not User B's
        contentsA = [m["data"]["content"] for m in stateA["memory"]]
        contentsB = [m["data"]["content"] for m in stateB["memory"]]

        assert "Hello from A" in contentsA
        assert "Hello from B" not in contentsA

        assert "Hello from B" in contentsB
        assert "Hello from A" not in contentsB


def test_concurrent_users_do_not_share_memory(test_config):
    """Two users messaging at the same time must not see each other's conversation (shared agent)."""
    import asyncio

    WhatsAppServer.create_app(MockWhatsAppAgent(latency=0.3), test_config)
    sent = {}

    async def fake_send(to, text):
        sent[to] = text

    async def _test():
        async def user(wa_id, text, delay):
            await asyncio.sleep(delay)
            await WhatsAppServer._handle_user_turn(sender_id=wa_id, text=text, wamid=f"wamid.{wa_id}")

        await asyncio.gather(user("1111", "Alice secret", 0.0), user("2222", "Bob says hi", 0.1))

    with patch.object(MetaWhatsAppClient, "send_text_message", side_effect=fake_send), \
         patch.object(MetaWhatsAppClient, "mark_as_read", new_callable=AsyncMock):
        asyncio.run(_test())

    assert sent == {"1111": "Bot reply to: Alice secret", "2222": "Bot reply to: Bob says hi"}

    store = SQLiteSessionStore(db_path=test_config.db_path)
    assert [m["data"]["content"] for m in store.load_session("1111")["memory"]] == \
        ["Alice secret", "Bot reply to: Alice secret"]
    assert [m["data"]["content"] for m in store.load_session("2222")["memory"]] == \
        ["Bob says hi", "Bot reply to: Bob says hi"]


def test_parallel_workers_serve_users_concurrently(test_config):
    """With max_workers > 1, users are served at the same time, still without sharing memory."""
    import asyncio

    test_config.max_workers = 2
    WhatsAppServer.create_app(MockWhatsAppAgent(latency=0.5), test_config)
    sent = {}

    async def fake_send(to, text):
        sent[to] = text

    async def _test():
        await asyncio.gather(
            WhatsAppServer._handle_user_turn(sender_id="1111", text="Alice secret", wamid="wamid.a"),
            WhatsAppServer._handle_user_turn(sender_id="2222", text="Bob says hi", wamid="wamid.b"),
        )

    with patch.object(MetaWhatsAppClient, "send_text_message", side_effect=fake_send), \
         patch.object(MetaWhatsAppClient, "mark_as_read", new_callable=AsyncMock):
        start = time.monotonic()
        asyncio.run(_test())
        elapsed = time.monotonic() - start

    assert sent == {"1111": "Bot reply to: Alice secret", "2222": "Bot reply to: Bob says hi"}
    assert elapsed < 0.9  # One at a time would take 1s


def test_parallel_workers_keep_each_user_in_order(test_config):
    """Quick messages from the same user are processed one at a time, each seeing the previous ones."""
    import asyncio

    test_config.max_workers = 3
    WhatsAppServer.create_app(MockWhatsAppAgent(latency=0.1), test_config)
    sent = []

    async def fake_send(to, text):
        sent.append(text)

    async def _test():
        await asyncio.gather(*(
            WhatsAppServer._handle_user_turn(sender_id="1111", text=f"m{i}", wamid=f"wamid.{i}") for i in range(3)))

    with patch.object(MetaWhatsAppClient, "send_text_message", side_effect=fake_send), \
         patch.object(MetaWhatsAppClient, "mark_as_read", new_callable=AsyncMock):
        asyncio.run(_test())

    assert sent == ["Bot reply to: m0", "Bot reply to: m0 | m1", "Bot reply to: m0 | m1 | m2"]


def test_finished_session_starts_over(test_config):
    """After the agent ends a conversation, the user's next message starts a new one."""
    import asyncio

    agent = MockWhatsAppAgent()
    WhatsAppServer.create_app(agent, test_config)
    store = SQLiteSessionStore(db_path=test_config.db_path)
    store.save_session("1111", {"version": 1, "finished": True, "orchestrators": [],
                                "memory": messages_to_dict([HumanMessage(content="Bye"), AIMessage(content="Bye!")])})

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock) as mock_send, \
         patch.object(MetaWhatsAppClient, "mark_as_read", new_callable=AsyncMock):
        asyncio.run(WhatsAppServer._handle_user_turn(sender_id="1111", text="Hi again", wamid="wamid.x"))

    mock_send.assert_awaited_once_with(to="1111", text="Bot reply to: Hi again")


def test_agent_serve_whatsapp_method_signature():
    """Verify serve_whatsapp is present on Agent and Server."""
    from sdialog.agents import Agent
    from sdialog.server import Server

    assert hasattr(Agent, "serve_whatsapp")
    assert callable(Agent.serve_whatsapp)

    assert hasattr(Server, "serve_whatsapp")
    assert callable(Server.serve_whatsapp)


# ---------------------------------------------------------------------------
# Test 6: Auto-Discovery & Webhook Sync (adapted from WAVoice)
# ---------------------------------------------------------------------------

def test_meta_discovery_inspect_and_discover():
    import asyncio
    from sdialog.whatsapp.discovery import MetaDiscoveryClient

    client = MetaDiscoveryClient(access_token="test_token", app_secret="test_secret")

    async def _test():
        # Mock inspect_token
        mock_token_data = {
            "app_id": "123456",
            "granular_scopes": [
                {
                    "scope": "whatsapp_business_management",
                    "target_ids": ["waba_101"],
                }
            ],
        }

        with patch.object(MetaDiscoveryClient, "inspect_token", new_callable=AsyncMock) as mock_inspect:
            mock_inspect.return_value = mock_token_data

            waba_id = await client.discover_waba_id()
            assert waba_id == "waba_101"

            # Mock list_phone_numbers
            with patch.object(MetaDiscoveryClient, "list_phone_numbers", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = [{"id": "phone_999", "display_phone_number": "+1555"}]

                phone_id = await client.discover_phone_number_id()
                assert phone_id == "phone_999"

    asyncio.run(_test())


def test_meta_discovery_configure_webhooks_mocked():
    import asyncio
    from sdialog.whatsapp.discovery import MetaDiscoveryClient

    client = MetaDiscoveryClient(access_token="test_token", app_secret="test_secret")

    async def _test():
        with patch.object(MetaDiscoveryClient, "inspect_token", new_callable=AsyncMock) as mock_inspect:
            mock_inspect.return_value = {"app_id": "app_555"}

            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"success": True}
                mock_post.return_value = mock_resp

                results = await client.configure_webhooks(
                    callback_url="https://test.ngrok.app/webhook",
                    verify_token="test_secret",
                    phone_number_id="phone_777",
                    waba_id="waba_888",
                )

                assert results["app_level"] is True
                assert results["waba_level"] is True
                assert results["phone_level"] is True
                assert mock_post.call_count == 3  # All 3 tiers synced

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Test 7: Per-user rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_per_minute(test_config):
    test_config.max_messages_per_minute = 3
    WhatsAppServer.create_app(MockWhatsAppAgent(), test_config)

    clock = {"now": 1000.0}
    with patch("time.monotonic", side_effect=lambda: clock["now"]):
        assert [WhatsAppServer._check_rate_limit("1111") for _ in range(3)] == [(True, None)] * 3
        allowed, notice = WhatsAppServer._check_rate_limit("1111")  # first rejection: notify
        assert not allowed and "too quickly" in notice
        assert WhatsAppServer._check_rate_limit("1111") == (False, None)  # then drop silently
        assert WhatsAppServer._check_rate_limit("2222") == (True, None)  # other users unaffected

        clock["now"] += 60
        assert WhatsAppServer._check_rate_limit("1111") == (True, None)  # window has slid


def test_rate_limit_per_hour(test_config):
    test_config.max_messages_per_minute = 6
    test_config.max_messages_per_hour = 10
    WhatsAppServer.create_app(MockWhatsAppAgent(), test_config)

    clock = {"now": 1000.0}
    with patch("time.monotonic", side_effect=lambda: clock["now"]):
        # One message every 30 seconds stays under the per-minute limit...
        for _ in range(10):
            assert WhatsAppServer._check_rate_limit("1111") == (True, None)
            clock["now"] += 30
        # ...but hits the per-hour one
        allowed, notice = WhatsAppServer._check_rate_limit("1111")
        assert not allowed and "message limit" in notice
        clock["now"] += 120
        assert WhatsAppServer._check_rate_limit("1111") == (False, None)  # one notice per hour

        clock["now"] += 3600
        assert WhatsAppServer._check_rate_limit("1111") == (True, None)


def test_rate_limit_disabled(test_config):
    test_config.max_messages_per_minute = None
    test_config.max_messages_per_hour = None
    WhatsAppServer.create_app(MockWhatsAppAgent(), test_config)
    assert all(WhatsAppServer._check_rate_limit("1111") == (True, None) for _ in range(100))


def test_rate_limit_drops_messages_without_calling_agent(test_config):
    test_config.max_messages_per_minute = 2
    agent = MockWhatsAppAgent()
    client = TestClient(WhatsAppServer.create_app(agent, test_config))

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock) as mock_send:
        for i in range(4):
            payload = json.dumps(_create_meta_payload(sender="1111", text=f"msg {i}", wamid=f"wamid.{i}")).encode()
            resp = client.post("/webhook", content=payload, headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _compute_hmac(payload, test_config.app_secret),
            })
            assert resp.status_code == 200

    assert agent.call_count == 2
    notices = [c for c in mock_send.await_args_list if "too quickly" in c.kwargs["text"]]
    assert len(notices) == 1


# ---------------------------------------------------------------------------
# Test 8: Session lifetime (new chats, "auto" timeout, retention)
# ---------------------------------------------------------------------------

def _post_signed(client, payload, secret):
    body = json.dumps(payload).encode()
    return client.post("/webhook", content=body, headers={
        "Content-Type": "application/json",
        "X-Hub-Signature-256": _compute_hmac(body, secret),
    })


def _set_last_active(db_path, wa_id, last_active):
    import sqlite3

    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("UPDATE whatsapp_sessions SET last_active = ? WHERE wa_id = ?", (last_active, wa_id))
    conn.close()


@pytest.mark.parametrize("value", [0, -5, "30m", True, 1.5])
def test_whatsapp_config_rejects_invalid_session_timeout(value):
    config = WhatsAppConfig(phone_number_id="123", access_token="token", app_secret="secret", session_timeout=value)
    with pytest.raises(ValueError, match="session_timeout"):
        config.validate()


@pytest.mark.parametrize("value", ["auto", None, 60])
def test_whatsapp_config_accepts_valid_session_timeout(value):
    WhatsAppConfig(phone_number_id="123", access_token="token", app_secret="secret",
                   session_timeout=value).validate()


def test_session_store_no_timeout(temp_db):
    store = SQLiteSessionStore(db_path=temp_db)
    store.save_session("1111", {"version": 1, "memory": []})
    _set_last_active(temp_db, "1111", "2000-01-01 00:00:00")

    assert store.load_session("1111", timeout_seconds=None) == {"version": 1, "memory": []}
    assert store.load_session("1111", timeout_seconds=1800) is None


def test_session_store_cleanup_inactive_sessions(temp_db):
    store = SQLiteSessionStore(db_path=temp_db)
    store.save_session("old", {"version": 1, "memory": []})
    store.save_session("recent", {"version": 1, "memory": []})
    _set_last_active(temp_db, "old", "2000-01-01 00:00:00")

    store.cleanup_old_records()  # Sessions are kept unless a retention period is given
    assert store.load_session("old", timeout_seconds=None) is not None

    store.cleanup_old_records(session_retention_days=30)
    assert store.load_session("old", timeout_seconds=None) is None
    assert store.load_session("recent", timeout_seconds=None) is not None


def test_cleanup_runs_at_most_once_per_interval(test_config):
    from sdialog.whatsapp import server as server_module

    WhatsAppServer.create_app(MockWhatsAppAgent(), test_config)
    with patch.object(SQLiteSessionStore, "cleanup_old_records") as mock_cleanup:
        WhatsAppServer._cleanup_if_due()
        assert mock_cleanup.call_count == 0  # Already ran in create_app

        WhatsAppServer._last_cleanup -= server_module.CLEANUP_INTERVAL
        WhatsAppServer._cleanup_if_due()
        WhatsAppServer._cleanup_if_due()
        mock_cleanup.assert_called_once_with(session_retention_days=test_config.session_retention_days)


def test_new_chat_resets_conversation_and_greets(test_config):
    """A request_welcome webhook (user opened a new chat) starts a fresh conversation with a greeting."""
    agent = MockWhatsAppAgent()
    client = TestClient(WhatsAppServer.create_app(agent, test_config))
    store = SQLiteSessionStore(db_path=test_config.db_path)
    store.save_session("1111", {"version": 1, "finished": False, "orchestrators": [],
                                "memory": messages_to_dict([HumanMessage(content="Old"), AIMessage(content="Old!")])})

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock) as mock_send:
        resp = _post_signed(client, _create_meta_payload(sender="1111", wamid="wamid.w", msg_type="request_welcome"),
                            test_config.app_secret)
        assert resp.status_code == 200
        mock_send.assert_awaited_once_with(to="1111", text="Bot greeting")

        _post_signed(client, _create_meta_payload(sender="1111", text="Hi", wamid="wamid.t"), test_config.app_secret)
        assert mock_send.await_args.kwargs["text"] == "Bot reply to: Hi"  # The old conversation is gone

    memory = messages_from_dict(store.load_session("1111")["memory"])
    assert [m.content for m in memory] == ["Bot greeting", "Hi", "Bot reply to: Hi"]


def test_auto_session_timeout_switches_on_new_chat_events(test_config):
    from sdialog.whatsapp.config import AUTO_SESSION_TIMEOUT, AUTO_SESSION_TIMEOUT_NEW_CHAT

    test_config.session_timeout = "auto"
    client = TestClient(WhatsAppServer.create_app(MockWhatsAppAgent(), test_config))
    assert WhatsAppServer._session_timeout() == AUTO_SESSION_TIMEOUT

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        _post_signed(client, _create_meta_payload(sender="1111", msg_type="request_welcome"), test_config.app_secret)
    assert WhatsAppServer._session_timeout() == AUTO_SESSION_TIMEOUT_NEW_CHAT

    # Explicit values are used as given
    for value in (60, None):
        test_config.session_timeout = value
        assert WhatsAppServer._session_timeout() == value


def test_meta_discovery_welcome_message_detection():
    import asyncio
    from sdialog.whatsapp.discovery import MetaDiscoveryClient

    client = MetaDiscoveryClient(access_token="test_token")
    cases = [
        (200, {"id": "phone_1", "conversational_automation": {"enable_welcome_message": True}}, True),
        (200, {"id": "phone_1"}, False),  # Field omitted when never configured
        (400, {"error": {"message": "Unsupported"}}, None),
    ]
    for status, body, expected in cases:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=status, json=MagicMock(return_value=body), text="")
            assert asyncio.run(client.is_welcome_message_enabled("phone_1")) is expected
            assert mock_get.call_args.kwargs["params"] == {"fields": "conversational_automation"}


@pytest.mark.parametrize("enabled", [True, False, None])
def test_serve_detects_welcome_messages(test_config, enabled):
    from sdialog.whatsapp.config import AUTO_SESSION_TIMEOUT, AUTO_SESSION_TIMEOUT_NEW_CHAT
    from sdialog.whatsapp.discovery import MetaDiscoveryClient

    test_config.session_timeout = "auto"
    with patch.object(MetaDiscoveryClient, "is_welcome_message_enabled", new_callable=AsyncMock,
                      return_value=enabled) as mock_detect, \
         patch("sdialog.whatsapp.server.uvicorn.run") as mock_run:
        WhatsAppServer.serve(MockWhatsAppAgent(), test_config)

    mock_detect.assert_awaited_once_with(test_config.phone_number_id)
    mock_run.assert_called_once()
    expected = AUTO_SESSION_TIMEOUT_NEW_CHAT if enabled else AUTO_SESSION_TIMEOUT
    assert WhatsAppServer._session_timeout() == expected


# ---------------------------------------------------------------------------
# Test 9: Stale messages (late webhook retries from Meta)
# ---------------------------------------------------------------------------

def test_stale_messages_are_ignored(test_config):
    """A message delivered long after it was sent (Meta retry) is not answered, and not answered on later retries."""
    agent = MockWhatsAppAgent()
    client = TestClient(WhatsAppServer.create_app(agent, test_config))

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock) as mock_send:
        stale = _create_meta_payload(sender="1111", text="Old", wamid="wamid.old", sent_seconds_ago=3600)
        for _ in range(2):  # The original late delivery, then a later retry of it
            assert _post_signed(client, stale, test_config.app_secret).status_code == 200
        for msg_type in ("audio", "request_welcome"):  # Not just text messages
            _post_signed(client, _create_meta_payload(sender="1111", wamid=f"wamid.{msg_type}", msg_type=msg_type,
                                                      sent_seconds_ago=3600), test_config.app_secret)
        mock_send.assert_not_awaited()
        assert agent.call_count == 0

        recent = _create_meta_payload(sender="1111", text="New", wamid="wamid.new", sent_seconds_ago=30)
        _post_signed(client, recent, test_config.app_secret)
    mock_send.assert_awaited_once_with(to="1111", text="Bot reply to: New")
    assert WhatsAppServer._store.is_message_processed("wamid.old")


def test_stale_messages_do_not_count_towards_rate_limits(test_config):
    test_config.max_messages_per_minute = 1
    agent = MockWhatsAppAgent()
    client = TestClient(WhatsAppServer.create_app(agent, test_config))

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        for i in range(3):
            _post_signed(client, _create_meta_payload(sender="1111", wamid=f"wamid.old{i}", sent_seconds_ago=600),
                         test_config.app_secret)
        _post_signed(client, _create_meta_payload(sender="1111", wamid="wamid.new"), test_config.app_secret)
    assert agent.call_count == 1


def test_max_message_age_none_answers_all(test_config):
    test_config.max_message_age = None
    agent = MockWhatsAppAgent()
    client = TestClient(WhatsAppServer.create_app(agent, test_config))

    with patch.object(MetaWhatsAppClient, "send_text_message", new_callable=AsyncMock):
        _post_signed(client, _create_meta_payload(sender="1111", sent_seconds_ago=86400), test_config.app_secret)
    assert agent.call_count == 1


def test_message_age():
    now = time.time()
    assert WhatsAppServer._message_age({"timestamp": str(int(now - 90))}) == pytest.approx(90, abs=2)
    assert WhatsAppServer._message_age({"timestamp": str(int(now + 30))}) == 0  # Clocks slightly out of sync
    assert WhatsAppServer._message_age({}) is None  # Missing or invalid timestamps: the message is answered
    assert WhatsAppServer._message_age({"timestamp": "soon"}) is None


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2m"])
def test_whatsapp_config_rejects_invalid_max_message_age(value):
    config = WhatsAppConfig(phone_number_id="123", access_token="token", app_secret="secret", max_message_age=value)
    with pytest.raises(ValueError, match="max_message_age"):
        config.validate()
