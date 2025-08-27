# =============================================================================
# tests/test_client.py
# =============================================================================
# Purpose:
# Comprehensive unit tests for the A2AClient class in client/client.py
# Tests cover initialization, task sending, task retrieval, error handling,
# and edge cases.
# =============================================================================

import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
import pytest

# Import the classes we're testing
from client.client import A2AClient, A2AClientHTTPError, A2AClientJSONError
from models.agent import AgentCapabilities, AgentCard, AgentSkill
from models.json_rpc import JSONRPCRequest
from models.request import GetTaskRequest, SendTaskRequest
from models.task import Message, Task, TaskSendParams, TaskStatus, TextPart


class TestA2AClient:
    """Test suite for the A2AClient class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.test_url = "http://test-agent.example.com"

        # Create a sample agent card for testing
        self.agent_card = AgentCard(
            id="test-agent-123",
            name="Test Agent",
            description="A test agent for unit testing",
            url=self.test_url,
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=True, pushNotifications=False, stateTransitionHistory=True
            ),
            skills=[
                AgentSkill(
                    id="test_skill",
                    name="Test Skill",
                    description="A skill for testing",
                    tags=["test", "demo"],
                    examples=["test this", "run test"],
                    inputModes=["text"],
                    outputModes=["text"],
                )
            ],
        )

        # Sample task data for testing
        self.sample_task_data = {
            "id": "task-123",
            "sessionId": "session-456",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello, test!"}],
            },
            "metadata": {"test": True},
        }

        # Sample task response
        self.sample_task = Task(
            id="task-123",
            status=TaskStatus(state="completed"),
            history=[
                Message(role="user", parts=[TextPart(text="Hello, test!")]),
                Message(
                    role="agent",
                    parts=[TextPart(text="Hello! This is a test response.")],
                ),
            ],
        )

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_with_agent_card(self):
        """Test client initialization with an agent card"""
        client = A2AClient(agent_card=self.agent_card)
        assert client.url == self.test_url

    def test_init_with_url(self):
        """Test client initialization with a direct URL"""
        client = A2AClient(url=self.test_url)
        assert client.url == self.test_url

    def test_init_without_agent_card_or_url(self):
        """Test that initialization fails without agent card or URL"""
        with pytest.raises(ValueError, match="Must provide either agent_card or url"):
            A2AClient()

    def test_init_with_both_agent_card_and_url(self):
        """Test that agent card takes precedence when both are provided"""
        client = A2AClient(agent_card=self.agent_card, url="http://different.url")
        assert client.url == self.test_url  # Should use agent_card.url

    # -------------------------------------------------------------------------
    # send_task Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_send_task_success(self):
        """Test successful task sending"""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": self.sample_task.model_dump(),
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            client = A2AClient(url=self.test_url)
            result = await client.send_task(self.sample_task_data)

            assert isinstance(result, Task)
            assert result.id == "task-123"
            assert result.status.state == "completed"

    @pytest.mark.asyncio
    async def test_send_task_creates_proper_request(self):
        """Test that send_task creates a properly formatted SendTaskRequest"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": self.sample_task.model_dump(),
        }
        mock_response.raise_for_status.return_value = None

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("uuid.uuid4") as mock_uuid,
        ):

            mock_uuid.return_value.hex = "mock-uuid-123"
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            client = A2AClient(url=self.test_url)
            await client.send_task(self.sample_task_data)

            # Verify the request was called with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            assert call_args[1]["json"]["method"] == "tasks/send"
            # assert call_args[1]['json']['id'] == 'mock-uuid-123'
            assert call_args[1]["json"]["jsonrpc"] == "2.0"
            assert "params" in call_args[1]["json"]
            assert call_args[1]["json"]["params"]["id"] == "task-123"

    @pytest.mark.asyncio
    async def test_send_task_http_error(self):
        """Test send_task handling of HTTP errors"""
        mock_response = Mock()
        mock_response.status_code = 500

        http_error = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=http_error
            )

            client = A2AClient(url=self.test_url)

            with pytest.raises(A2AClientHTTPError):
                await client.send_task(self.sample_task_data)

    @pytest.mark.asyncio
    async def test_send_task_json_decode_error(self):
        """Test send_task handling of JSON decode errors"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            client = A2AClient(url=self.test_url)

            with pytest.raises(A2AClientJSONError):
                await client.send_task(self.sample_task_data)

    # -------------------------------------------------------------------------
    # get_task Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_task_success(self):
        """Test successful task retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": self.sample_task.model_dump(),
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            client = A2AClient(url=self.test_url)
            result = await client.get_task({"id": "task-123"})

            assert isinstance(result, Task)
            assert result.id == "task-123"

    @pytest.mark.asyncio
    async def test_get_task_creates_proper_request(self):
        """Test that get_task creates a properly formatted GetTaskRequest"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": self.sample_task.model_dump(),
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            client = A2AClient(url=self.test_url)
            await client.get_task({"id": "task-123", "historyLength": 10})

            # Verify the request was called with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            assert call_args[1]["json"]["method"] == "tasks/get"
            assert "params" in call_args[1]["json"]

    # -------------------------------------------------------------------------
    # _send_request Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_send_request_timeout(self):
        """Test that _send_request uses correct timeout"""
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            client = A2AClient(url=self.test_url)
            request = SendTaskRequest(
                id="test-id", params=TaskSendParams(**self.sample_task_data)
            )

            await client._send_request(request)

            # Verify timeout was set correctly
            call_args = mock_post.call_args
            assert call_args[1]["timeout"] == 60

    @pytest.mark.asyncio
    async def test_send_request_proper_url_and_json(self):
        """Test that _send_request sends to correct URL with proper JSON"""
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            client = A2AClient(url=self.test_url)
            request = SendTaskRequest(
                id="test-id", params=TaskSendParams(**self.sample_task_data)
            )

            await client._send_request(request)

            # Verify URL and JSON payload
            call_args = mock_post.call_args
            assert call_args[0][0] == self.test_url  # First positional arg is URL
            assert "json" in call_args[1]
            assert call_args[1]["json"] == request.model_dump()

    # -------------------------------------------------------------------------
    # Error Class Tests
    # -------------------------------------------------------------------------

    def test_a2a_client_http_error(self):
        """Test A2AClientHTTPError exception"""
        error = A2AClientHTTPError(500, "Server Error")
        assert str(error) == "(500, 'Server Error')"

    def test_a2a_client_json_error(self):
        """Test A2AClientJSONError exception"""
        error = A2AClientJSONError("Invalid JSON format")
        assert str(error) == "Invalid JSON format"

    # -------------------------------------------------------------------------
    # Integration-like Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_end_to_end_task_workflow(self):
        """Test a complete workflow of sending and getting a task"""
        # Mock responses for both send and get operations
        send_response = Mock()
        send_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "send-id",
            "result": self.sample_task.model_dump(),
        }
        send_response.raise_for_status.return_value = None

        get_response = Mock()
        get_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "get-id",
            "result": self.sample_task.model_dump(),
        }
        get_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(side_effect=[send_response, get_response])
            mock_client.return_value.__aenter__.return_value.post = mock_post

            client = A2AClient(agent_card=self.agent_card)

            # Send task
            sent_task = await client.send_task(self.sample_task_data)
            assert isinstance(sent_task, Task)
            assert sent_task.id == "task-123"

            # Get task
            retrieved_task = await client.get_task({"id": "task-123"})
            assert isinstance(retrieved_task, Task)
            assert retrieved_task.id == "task-123"

            # Verify both calls were made
            assert mock_post.call_count == 2


# =============================================================================
# Pytest Configuration and Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_card():
    """Fixture providing a sample AgentCard for tests"""
    return AgentCard(
        id="fixture-agent",
        name="Fixture Agent",
        description="An agent for testing",
        url="http://fixture.example.com",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        skills=[],
    )


@pytest.fixture
def sample_task_payload():
    """Fixture providing sample task payload data"""
    return {
        "id": "fixture-task",
        "sessionId": "fixture-session",
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Fixture message"}],
        },
    }


# =============================================================================
# Parametrized Tests for Edge Cases
# =============================================================================


class TestA2AClientEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.parametrize(
        "invalid_url", ["", None, "not-a-url", "ftp://invalid-protocol.com"]
    )
    def test_client_with_invalid_urls(self, invalid_url):
        """Test client behavior with various invalid URLs"""
        # Note: The client doesn't validate URL format in __init__,
        # so this will only fail when actually making requests
        if invalid_url is None or invalid_url == "":
            with pytest.raises(ValueError):
                A2AClient(url=invalid_url)
        else:
            client = A2AClient(url=invalid_url)
            assert client.url == invalid_url

    @pytest.mark.parametrize("empty_payload", [{}, {"id": ""}, {"message": None}])
    @pytest.mark.asyncio
    async def test_send_task_with_invalid_payloads(self, empty_payload):
        """Test send_task with various invalid payloads"""
        client = A2AClient(url="http://test.com")

        # These should raise validation errors from Pydantic models
        with pytest.raises((ValueError, TypeError)):
            await client.send_task(empty_payload)


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__])
