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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from sdialog.whatsapp import (
    WhatsAppConfig,
    WhatsAppServer,
    SQLiteSessionStore,
    MetaWhatsAppClient,
)
from sdialog.whatsapp.session_store import restore_agent_state, dump_agent_state


class MockWhatsAppAgent:
    """Mock agent for WhatsApp testing."""

    def __init__(self, name="TestBot"):
        self.name = name
        self.memory = [SystemMessage(content="You are a helpful test bot.")]
        self.call_count = 0
        self.last_input = None

    def __call__(self, message: str, **kwargs) -> str:
        self.call_count += 1
        self.last_input = message
        self.memory.append(HumanMessage(content=message))
        reply = f"Bot reply to: {message}"
        self.memory.append(AIMessage(content=reply))
        return reply

    def reset(self, seed=None):
        self.memory = [SystemMessage(content="You are a helpful test bot.")]
        self.call_count = 0


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


def test_whatsapp_config_validation():
    config = WhatsAppConfig(phone_number_id=None, access_token=None)
    with pytest.raises(ValueError, match="phone_number_id is required"):
        config.validate(dry_run=False)

    config = WhatsAppConfig(phone_number_id="123", access_token=None)
    with pytest.raises(ValueError, match="access_token is required"):
        config.validate(dry_run=False)

    # Valid config
    config = WhatsAppConfig(phone_number_id="123", access_token="token")
    config.validate(dry_run=False)  # Should not raise


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


def test_state_bridge_restore_and_dump():
    agent = MockWhatsAppAgent()
    agent.memory.append(HumanMessage(content="User 1"))
    agent.memory.append(AIMessage(content="Bot 1"))

    # Dump state
    state = dump_agent_state(agent, max_history_turns=10)
    assert len(state["memory"]) == 2
    assert state["memory"][0]["content"] == "User 1"
    assert state["memory"][1]["content"] == "Bot 1"

    # Restore on a fresh agent
    new_agent = MockWhatsAppAgent()
    restore_agent_state(new_agent, state)
    assert len(new_agent.memory) == 3  # SystemMessage + 2 turns
    assert isinstance(new_agent.memory[1], HumanMessage)
    assert new_agent.memory[1].content == "User 1"


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
        contentsA = [m["content"] for m in stateA["memory"]]
        contentsB = [m["content"] for m in stateB["memory"]]

        assert "Hello from A" in contentsA
        assert "Hello from B" not in contentsA

        assert "Hello from B" in contentsB
        assert "Hello from A" not in contentsB


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
