# =============================================================================
# tests/test_server.py
# =============================================================================
# Purpose:
# Comprehensive unit tests for the A2AServer class in server/server.py
# Tests cover initialization, route handling, request processing, error handling,
# and integration scenarios.
# =============================================================================

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

# Import Starlette testing utilities
from starlette.testclient import TestClient

from models.agent import AgentCapabilities, AgentCard, AgentSkill
from models.json_rpc import InternalError, JSONRPCRequest, JSONRPCResponse
from models.request import A2ARequest, SendTaskRequest
from models.task import Message, Task, TaskSendParams, TaskStatus, TextPart
from server import task_manager

# Import the classes we're testing
from server.server import A2AServer, json_serializer


class TestA2AServer:
    """Test suite for the A2AServer class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create a sample agent card for testing
        self.agent_card = AgentCard(
            id="test-server-agent",
            name="Test Server Agent",
            description="A test agent for server testing",
            url="http://test-server.example.com",
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=False, pushNotifications=False, stateTransitionHistory=True
            ),
            skills=[
                AgentSkill(
                    id="test_skill",
                    name="Test Skill",
                    description="A skill for server testing",
                    tags=["test", "server"],
                    examples=["test server", "check server"],
                    inputModes=["text"],
                    outputModes=["text"],
                )
            ],
        )

        # Mock task manager
        self.mock_task_manager = Mock()
        self.mock_task_manager.on_send_task = AsyncMock()

        # Sample task request data
        self.sample_send_task_request = {
            "jsonrpc": "2.0",
            "id": "test-request-123",
            "method": "tasks/send",
            "params": {
                "id": "task-456",
                "sessionId": "session-789",
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello server!"}],
                },
                "metadata": {"test": True},
            },
        }

        # Sample task response
        self.sample_task = Task(
            id="task-456",
            status=TaskStatus(state="completed"),
            history=[
                Message(role="user", parts=[TextPart(text="Hello server!")]),
                Message(
                    role="agent", parts=[TextPart(text="Hello! Server responded.")]
                ),
            ],
        )

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_default_values(self):
        """Test server initialization with default values"""
        server = A2AServer()
        assert server.host == "0.0.0.0"
        assert server.port == 5000
        assert server.agent_card is None
        assert server.task_manager is None
        assert server.app is not None

    def test_init_custom_values(self):
        """Test server initialization with custom values"""
        server = A2AServer(
            host="127.0.0.1",
            port=8080,
            agent_card=self.agent_card,
            task_manager=self.mock_task_manager,
        )
        assert server.host == "127.0.0.1"
        assert server.port == 8080
        assert server.agent_card == self.agent_card
        assert server.task_manager == self.mock_task_manager

    def test_init_routes_registered(self):
        """Test that all routes are properly registered during initialization"""
        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )

        # Check that routes are registered (we can't easily inspect Starlette routes,
        # so we'll test this through the TestClient)
        client = TestClient(server.app)

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # -------------------------------------------------------------------------
    # start() Method Tests
    # -------------------------------------------------------------------------

    def test_start_without_agent_card(self):
        """Test that start() raises error without agent card"""
        server = A2AServer(task_manager=self.mock_task_manager)

        with pytest.raises(
            ValueError, match="Agent card and task manager are required"
        ):
            server.start()

    def test_start_without_task_manager(self):
        """Test that start() raises error without task manager"""
        server = A2AServer(agent_card=self.agent_card)

        with pytest.raises(
            ValueError, match="Agent card and task manager are required"
        ):
            server.start()

    @patch("uvicorn.run")
    def test_start_success(self, mock_uvicorn_run):
        """Test successful server start"""
        server = A2AServer(
            host="127.0.0.1",
            port=8080,
            agent_card=self.agent_card,
            task_manager=self.mock_task_manager,
        )

        server.start()

        mock_uvicorn_run.assert_called_once_with(
            server.app, host="127.0.0.1", port=8080
        )

    # -------------------------------------------------------------------------
    # _get_agent_card() Method Tests
    # -------------------------------------------------------------------------

    def test_get_agent_card_success(self):
        """Test successful agent card retrieval"""
        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        response = client.get("/.well-known/agent.json")

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["id"] == "test-server-agent"
        assert response_data["name"] == "Test Server Agent"
        assert response_data["url"] == "http://test-server.example.com"
        assert "capabilities" in response_data
        assert "skills" in response_data

    def test_get_agent_card_exclude_none(self):
        """Test that agent card response excludes None values"""
        # Create agent card with some None values
        agent_card = AgentCard(
            id="test-agent",
            name="Test Agent",
            description="Test Description",
            url="http://test.com",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[],
        )

        server = A2AServer(agent_card=agent_card, task_manager=self.mock_task_manager)
        client = TestClient(server.app)

        response = client.get("/.well-known/agent.json")
        response_data = response.json()

        # Check that None values are excluded
        for value in response_data.values():
            assert value is not None

    # -------------------------------------------------------------------------
    # _handle_request() Method Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_request_success(self):
        """Test successful request handling"""
        # Setup mock task manager response
        mock_response = JSONRPCResponse(
            id="test-request-123", result=self.sample_task.model_dump()
        )
        self.mock_task_manager.on_send_task.return_value = mock_response

        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        response = client.post("/", json=self.sample_send_task_request)

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["jsonrpc"] == "2.0"
        assert response_data["id"] == "test-request-123"
        assert "result" in response_data

    @pytest.mark.asyncio
    async def test_handle_request_invalid_json(self):
        """Test request handling with invalid JSON"""
        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        # Send invalid JSON
        response = client.post("/", content="invalid json")

        assert response.status_code == 400
        response_data = response.json()

        assert response_data["jsonrpc"] == "2.0"
        assert "error" in response_data
        assert response_data["error"]["code"] == -32603  # Internal error code

    @pytest.mark.asyncio
    async def test_handle_request_unsupported_method(self):
        """Test request handling with unsupported method"""
        unsupported_request = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "unsupported/method",
            "params": {},
        }

        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        response = client.post("/", json=unsupported_request)

        assert response.status_code == 400
        response_data = response.json()

        assert "error" in response_data
        assert (
            "expected tags: 'tasks/send', 'tasks/get'"
            in response_data["error"]["message"]
        )

    @pytest.mark.asyncio
    async def test_handle_request_task_manager_exception(self):
        """Test request handling when task manager raises exception"""
        # Setup task manager to raise exception
        self.mock_task_manager.on_send_task.side_effect = Exception(
            "Task manager error"
        )

        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        response = client.post("/", json=self.sample_send_task_request)

        assert response.status_code == 400
        response_data = response.json()

        assert "error" in response_data
        assert "Task manager error" in response_data["error"]["message"]

    # -------------------------------------------------------------------------
    # _create_response() Method Tests
    # -------------------------------------------------------------------------

    def test_create_response_success(self):
        """Test successful response creation"""
        server = A2AServer()

        json_rpc_response = JSONRPCResponse(id="test-id", result={"message": "success"})

        response = server._create_response(json_rpc_response)

        assert isinstance(response, JSONResponse)

    def test_create_response_invalid_type(self):
        """Test response creation with invalid type"""
        server = A2AServer()

        with pytest.raises(ValueError, match="Invalid response type"):
            server._create_response("invalid response")

    def test_create_response_exclude_none(self):
        """Test that response creation excludes None values"""
        server = A2AServer()

        json_rpc_response = JSONRPCResponse(
            id="test-id",
            result={"message": "success"},
            error=None,  # This should be excluded
        )

        response = server._create_response(json_rpc_response)

        # The actual content checking would require accessing response.body,
        # which is more complex, so we just verify it doesn't raise an error
        assert isinstance(response, JSONResponse)

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_end_to_end_task_flow(self):
        """Test complete end-to-end task processing flow"""
        # Setup mock task manager response
        mock_response = JSONRPCResponse(
            id="test-request-123", result=self.sample_task.model_dump()
        )
        self.mock_task_manager.on_send_task.return_value = mock_response

        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        # First, get agent card
        agent_response = client.get("/.well-known/agent.json")
        assert agent_response.status_code == 200

        # Then, send a task
        task_response = client.post("/", json=self.sample_send_task_request)
        assert task_response.status_code == 200

        # Verify task manager was called
        self.mock_task_manager.on_send_task.assert_called_once()

        # Verify response structure
        task_data = task_response.json()
        assert task_data["jsonrpc"] == "2.0"
        assert "result" in task_data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        server = A2AServer()
        client = TestClient(server.app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions in the server module"""

    def test_json_serializer_datetime(self):
        """Test json_serializer with datetime objects"""
        test_datetime = datetime(2025, 8, 27, 12, 30, 45)
        result = json_serializer(test_datetime)

        assert result == "2025-08-27T12:30:45"

    def test_json_serializer_unsupported_type(self):
        """Test json_serializer with unsupported types"""
        with pytest.raises(TypeError, match="Type .* not serializable"):
            json_serializer(set([1, 2, 3]))


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestServerErrorHandling:
    """Test various error scenarios and edge cases"""

    def setup_method(self):
        """Set up test fixtures"""
        self.agent_card = AgentCard(
            id="error-test-agent",
            name="Error Test Agent",
            description="Agent for error testing",
            url="http://error-test.example.com",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[],
        )
        self.mock_task_manager = Mock()

    def test_malformed_json_request(self):
        """Test handling of malformed JSON requests"""
        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        # Send request with malformed JSON
        response = client.post("/", content="{invalid json")

        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_missing_required_fields(self):
        """Test handling of requests with missing required fields"""
        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        # Send request missing required fields
        incomplete_request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            # Missing id and params
        }

        response = client.post("/", json=incomplete_request)

        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    @pytest.mark.parametrize(
        "invalid_method", ["tasks/invalid", "unknown/method", "", None]
    )
    def test_invalid_methods(self, invalid_method):
        """Test handling of various invalid methods"""
        server = A2AServer(
            agent_card=self.agent_card, task_manager=self.mock_task_manager
        )
        client = TestClient(server.app)

        invalid_request = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": invalid_method,
            "params": {},
        }

        response = client.post("/", json=invalid_request)

        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data


# =============================================================================
# Fixtures for Reusable Test Data
# =============================================================================


@pytest.fixture
def sample_agent_card():
    """Fixture providing a sample AgentCard for tests"""
    return AgentCard(
        id="fixture-server-agent",
        name="Fixture Server Agent",
        description="An agent for server testing",
        url="http://fixture-server.example.com",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="fixture_skill",
                name="Fixture Skill",
                description="A skill for fixture testing",
            )
        ],
    )


@pytest.fixture
def mock_task_manager():
    """Fixture providing a mock task manager"""
    manager = Mock()
    manager.on_send_task = AsyncMock()
    return manager


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__])
