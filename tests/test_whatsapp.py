# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
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

    def __call__(self, message: str, **kwargs) -> str:
        self.call_count += 1
        self.last_input = message
        self.memory.append(HumanMessage(content=message))
        time.sleep(self.latency)
        seen = [m.content for m in self.memory if isinstance(m, HumanMessage)]
        reply = f"Bot reply to: {' | '.join(seen)}"
        self.memory.append(AIMessage(content=reply))
        return reply

    def reset(self, seed=None):
        self.memory[:] = self.memory[:1]
        self._finished = False

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
    )


# ---------------------------------------------------------------------------
# Test 1: Config
# ---------------------------------------------------------------------------

def test_whatsapp_config_defaults():
    config = WhatsAppConfig(verify_token="custom_token")
    assert config.port == 1234
    assert config.verify_token == "custom_token"
    assert config.session_timeout == 1800
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

def _create_meta_payload(sender="15551234567", text="Hello agent", wamid="wamid.123"):
    """Helper to create a standard Meta WhatsApp webhook payload."""
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
                            "messages": [
                                {
                                    "from": sender,
                                    "id": wamid,
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
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
