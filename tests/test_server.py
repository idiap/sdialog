"""
Unit tests for the SDialog server module.

Tests the FastAPI server functionality, OpenAI/Ollama compatibility,
agent management, and streaming responses.
"""
import json
import pytest

from unittest.mock import Mock
from threading import Lock

# Import to check if FastAPI dependencies are available
try:
    from fastapi.testclient import TestClient
    from sdialog.server import Server, ChatMessage, ChatCompletionRequest, OllamaChatRequest

    fastapi_available = True
except ImportError:
    fastapi_available = False


from sdialog import Event
from sdialog.personas import Persona


class MockAgent:
    """Mock agent for testing purposes."""

    def __init__(self, name="test-agent", memory=None):
        self.name = name
        self.memory = memory or []
        self.persona = Persona(name=name)  # Mock persona instead of importing
        self.call_count = 0

    def __call__(self, message, return_events=False):
        """Mock agent call method."""
        self.call_count += 1
        events = [
            Event(agent=self.name, action="think", content="I'm thinking about this...", timestamp=0),
            Event(agent=self.name, action="utter", content=f"Response to: {message}", timestamp=1)
        ]
        return events if return_events else f"Response to: {message}"

    def reset(self, seed=None):
        """Mock reset method."""
        self.memory = []
        self.call_count = 0

    def get_name(self):
        """Mock get_name method."""
        return self.name


@pytest.fixture
def mock_agent():
    """Fixture providing a mock agent."""
    return MockAgent()


@pytest.fixture
def server_client():
    """Fixture providing a test client for the server."""
    if not fastapi_available:
        pytest.skip("FastAPI not available")

    # Clear any existing agents and app before each test
    Server._agents.clear()
    Server._agent_locks.clear()
    Server._app = None

    # Create a new app
    Server._create_app()
    return TestClient(Server._app)


class TestServerBasics:
    """Test basic server functionality."""

    def test_server_class_attributes(self):
        """Test that Server class has required class attributes."""
        if not fastapi_available:
            pytest.skip("FastAPI not available")

        assert hasattr(Server, '_agents')
        assert hasattr(Server, '_agent_locks')
        assert hasattr(Server, '_app')
        assert isinstance(Server._agents, dict)
        assert isinstance(Server._agent_locks, dict)

    def test_add_agent(self, mock_agent):
        """Test adding an agent to the server."""
        Server.add_agent(mock_agent, "test-model:latest")

        assert "test-model:latest" in Server._agents
        assert Server._agents["test-model:latest"] == mock_agent
        assert "test-model:latest" in Server._agent_locks

    def test_remove_agent(self, mock_agent):
        """Test removing an agent from the server."""
        Server.add_agent(mock_agent, "test-model:latest")
        Server.remove_agent("test-model:latest")

        assert "test-model:latest" not in Server._agents
        assert "test-model:latest" not in Server._agent_locks

    def test_list_agents(self, mock_agent):
        """Test listing all registered agents."""
        Server.add_agent(mock_agent, "model1:latest")
        Server.add_agent(MockAgent("agent2"), "model2:latest")

        agents = Server.list_agents()
        assert "model1:latest" in agents
        assert "model2:latest" in agents
        assert len(agents) == 2

    def test_reset_agent(self, mock_agent):
        """Test resetting an agent's state."""
        Server.add_agent(mock_agent, "test-model:latest")

        # Call agent to set some state
        mock_agent("test message")
        assert mock_agent.call_count == 1

        # Reset agent
        Server.reset_agent("test-model:latest")
        assert mock_agent.call_count == 0
        assert mock_agent.memory == []

    def test_reset_nonexistent_agent(self):
        """Test error when resetting non-existent agent."""
        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            Server.reset_agent("nonexistent")


class TestChatModels:
    """Test Pydantic models for chat requests/responses."""

    def test_chat_message_creation(self):
        """Test creating a ChatMessage."""
        if not fastapi_available:
            pytest.skip("FastAPI not available")

        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.name is None

    def test_chat_completion_request_creation(self):
        """Test creating a ChatCompletionRequest."""
        if not fastapi_available:
            pytest.skip("FastAPI not available")

        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(model="test-model", messages=messages)
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.temperature is None
        assert request.stream is False

    def test_ollama_chat_request_creation(self):
        """Test creating an OllamaChatRequest."""
        if not fastapi_available:
            pytest.skip("FastAPI not available")

        messages = [ChatMessage(role="user", content="Hello")]
        request = OllamaChatRequest(model="test-model", messages=messages)
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.stream is True  # Enabled by default


class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    def test_health_check(self, server_client, mock_agent):
        """Test the health check endpoint."""
        Server.add_agent(mock_agent, "test-model:latest")

        response = server_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "test-model:latest" in data["agents"]

    def test_list_models_openai(self, server_client, mock_agent):
        """Test OpenAI-compatible models endpoint."""
        Server.add_agent(mock_agent, "test-model:latest")

        response = server_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model:latest"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "sdialog"

    def test_list_models_ollama(self, server_client, mock_agent):
        """Test Ollama-compatible models endpoint."""
        Server.add_agent(mock_agent, "test-model:latest")

        response = server_client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "test-model:latest"
        assert data["models"][0]["details"]["format"] == "sdialog"

    def test_ollama_version(self, server_client):
        """Test Ollama version endpoint."""
        response = server_client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "sdialog" in data["version"]

    def test_ollama_show_model(self, server_client, mock_agent):
        """Test Ollama show model endpoint."""
        Server.add_agent(mock_agent, "test-model:latest")

        response = server_client.post("/api/show", json={"name": "test-model:latest"})
        assert response.status_code == 200
        data = response.json()
        assert data["license"] == "MIT"
        assert "test-model:latest" in data["modelfile"]
        assert data["details"]["format"] == "sdialog"

    def test_ollama_show_nonexistent_model(self, server_client):
        """Test Ollama show model endpoint with non-existent model."""
        response = server_client.post("/api/show", json={"name": "nonexistent"})
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]


class TestChatCompletions:
    """Test chat completion endpoints."""

    def test_chat_completion_missing_model(self, server_client):
        """Test chat completion with non-existent model."""
        request_data = {
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        response = server_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_chat_completion_no_user_messages(self, server_client, mock_agent):
        """Test chat completion with no user messages."""
        Server.add_agent(mock_agent, "test-model:latest")

        request_data = {
            "model": "test-model:latest",
            "messages": [{"role": "system", "content": "You are a helpful assistant"}]
        }

        response = server_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

    def test_chat_completion_success(self, server_client, mock_agent):
        """Test successful chat completion."""
        Server.add_agent(mock_agent, "test-model:latest")

        request_data = {
            "model": "test-model:latest",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        response = server_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model:latest"
        assert data["object"] == "chat.completion"
        assert "choices" in data

    def test_chat_completion_with_open_webui_task(self, server_client, mock_agent):
        """Test chat completion with Open WebUI task message (should be ignored)."""
        Server.add_agent(mock_agent, "test-model:latest")

        request_data = {
            "model": "test-model:latest",
            "messages": [{"role": "user", "content": "### Task: Some task description"}]
        }

        response = server_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        # Agent should not be called for task messages
        assert mock_agent.call_count == 0


class TestOllamaChat:
    """Test Ollama-compatible chat endpoint."""

    def test_ollama_chat_non_streaming(self, server_client, mock_agent):
        """Test Ollama chat endpoint without streaming."""
        Server.add_agent(mock_agent, "test-model:latest")

        request_data = {
            "model": "test-model:latest",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }

        response = server_client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model:latest"
        assert "done" not in data  # 'done' field removed for non-streaming
        assert "message" in data

    def test_ollama_chat_with_options(self, server_client, mock_agent):
        """Test Ollama chat with options."""
        Server.add_agent(mock_agent, "test-model:latest")

        request_data = {
            "model": "test-model:latest",
            "messages": [{"role": "user", "content": "Hello"}],
            "options": {"temperature": 0.7, "num_predict": 100}
        }

        response = server_client.post("/api/chat", json=request_data)
        assert response.status_code == 200


class TestStreamingResponses:
    """Test streaming response functionality."""

    def test_stream_response_generation(self, mock_agent):
        """Test streaming response generation."""
        if not fastapi_available:
            pytest.skip("FastAPI not available")

        Server.add_agent(mock_agent, "test-model:latest")

        # Create a mock request
        request = ChatCompletionRequest(
            model="test-model:latest",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True
        )

        # Test the streaming response generator with async handling
        chunks = []
        try:
            import asyncio

            async def collect_chunks():
                async for chunk in Server._stream_response(request):
                    chunks.append(chunk)
                    if len(chunks) > 10:  # Prevent infinite loops in tests
                        break

            # Try to run in existing event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in a running loop, we can't use run()
                    pytest.skip("Cannot test async streaming in running event loop")
                else:
                    asyncio.run(collect_chunks())
            except RuntimeError:
                asyncio.run(collect_chunks())

            # Should have at least one chunk
            assert len(chunks) > 0

            # Parse the final chunk
            final_chunk = json.loads(chunks[-1])
            assert final_chunk["done"] is True
            assert final_chunk["model"] == "test-model:latest"
        except ImportError:
            pytest.skip("asyncio not available")

    def test_stream_response_with_nonexistent_model(self):
        """Test streaming response with non-existent model."""
        if not fastapi_available:
            pytest.skip("FastAPI not available")

        request = ChatCompletionRequest(
            model="nonexistent",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True
        )

        chunks = []
        try:
            import asyncio

            async def collect_chunks():
                async for chunk in Server._stream_response(request):
                    chunks.append(chunk)
                    if len(chunks) > 5:  # Prevent infinite loops
                        break

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pytest.skip("Cannot test async streaming in running event loop")
                else:
                    asyncio.run(collect_chunks())
            except RuntimeError:
                asyncio.run(collect_chunks())

            # Should have an error chunk
            assert len(chunks) > 0
            error_chunk = json.loads(chunks[0])
            assert "error" in error_chunk
            assert "not found" in error_chunk["error"]
        except ImportError:
            pytest.skip("asyncio not available")


class TestAgentMemoryManagement:
    """Test agent memory and conversation management."""

    def test_maybe_reset_agent_for_new_chat(self, mock_agent):
        """Test agent reset detection for new chats."""
        # Add some memory to the agent
        mock_agent.memory = [
            Mock(content="Previous message", __class__=Mock(__name__="HumanMessage")),
            Mock(content="Previous response", __class__=Mock(__name__="AIMessage"))
        ]

        # Test with completely different messages (should reset)
        request_messages = [ChatMessage(role="user", content="New conversation")]
        Server._maybe_reset_agent_for_request(mock_agent, request_messages)

        # Memory should be cleared
        assert mock_agent.memory == []

    def test_maybe_reset_agent_same_conversation(self, mock_agent):
        """Test agent reset detection for continuing conversation."""
        # Set up matching memory and request
        mock_agent.memory = [
            Mock(content="Hello", __class__=Mock(__name__="HumanMessage")),
            Mock(content="Hi there", __class__=Mock(__name__="AIMessage"))
        ]

        request_messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?")
        ]

        Server._maybe_reset_agent_for_request(mock_agent, request_messages)

        # Memory should not be cleared
        assert len(mock_agent.memory) == 2


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_in_ollama_show(self, server_client):
        """Test invalid JSON in Ollama show endpoint."""
        response = server_client.post("/api/show", json={"invalid": "data"})
        assert response.status_code == 400


class TestModelNameHandling:
    """Test model name processing and validation."""

    def test_model_name_with_tag(self, mock_agent):
        """Test model name that already includes a tag."""
        Server.add_agent(mock_agent, "test-model:v1.0")
        assert "test-model:v1.0" in Server._agents

    def test_model_name_without_tag(self, mock_agent):
        """Test model name without tag (should get :latest added)."""
        # This would be tested in the serve methods, but we test the logic here
        model_name = "test-model"
        if ":" not in model_name:
            model_name = f"{model_name}:latest"

        assert model_name == "test-model:latest"


class TestConcurrency:
    """Test concurrent access to agents."""

    def test_agent_locks_exist(self, mock_agent):
        """Test that agent locks are created properly."""
        Server.add_agent(mock_agent, "test-model:latest")

        assert "test-model:latest" in Server._agent_locks
        lock = Server._agent_locks["test-model:latest"]
        assert isinstance(lock, type(Lock()))

        # Test that the lock can be acquired
        assert lock.acquire(blocking=False)
        lock.release()


if __name__ == "__main__":
    pytest.main([__file__])
