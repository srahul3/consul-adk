import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from utilities.consul_agent import ConsulEnabledAIAgent

class DummyDiscovery:
    async def get_kv_variable(self, key):
        if key == "description":
            return "Test description"
        if key == "instruction":
            return "Test instruction with more than 20 chars."
        return None

class DummyAgent(ConsulEnabledAIAgent):
    def build_agent(self):
        return MagicMock(name="dummy_agent")

def test_get_llm_tools():
    agent = DummyAgent(DummyDiscovery())
    agent._list_agents = MagicMock()
    agent._agent_skills = MagicMock()
    agent._delegate_task = MagicMock()
    agent._tell_time = MagicMock()
    mock_user_tool_1 = MagicMock()
    mock_user_tool_2 = MagicMock()
    agent._user_defined_tools = {
        "tool1": mock_user_tool_1,
        "tool2": mock_user_tool_2
    }
    mock_mcp_tool_1 = MagicMock()
    mock_mcp_tool_2 = MagicMock()
    agent._remote_mcp_tools = {
        "mcp1": mock_mcp_tool_1,
        "mcp2": mock_mcp_tool_2
    }
    agent.is_orchestrator = True
    tools = agent._get_llm_tools()
    assert mock_user_tool_1 in tools
    assert mock_user_tool_2 in tools
    assert mock_mcp_tool_1 in tools
    assert mock_mcp_tool_2 in tools
    assert any(callable(t) for t in tools)
    agent.is_orchestrator = False
    tools = agent._get_llm_tools()
    assert mock_user_tool_1 in tools
    assert mock_user_tool_2 in tools
    assert mock_mcp_tool_1 in tools
    assert mock_mcp_tool_2 in tools
    assert not any(t == agent._list_agents for t in tools)

@pytest.mark.asyncio
def test_description():
    agent = DummyAgent(DummyDiscovery())
    desc = asyncio.run(agent._description())
    assert desc == "Test description"

@pytest.mark.asyncio
def test_description_default():
    class DummyDiscoveryNone:
        async def get_kv_variable(self, key):
            return None
    agent = DummyAgent(DummyDiscoveryNone())
    desc = asyncio.run(agent._description())
    assert desc == "Default agent description."

@pytest.mark.asyncio
def test_root_instruction():
    agent = DummyAgent(DummyDiscovery())
    context = MagicMock()
    instr = asyncio.run(agent._root_instruction(context))
    assert "Test instruction" in instr
    assert "Available agents:" in instr
    assert "Agent skills" in instr

@pytest.mark.asyncio
def test_delegate_task():
    agent = DummyAgent(DummyDiscovery())
    agent.connectors = {"foo": MagicMock()}
    agent.connectors["foo"].send_task = AsyncMock(return_value=MagicMock(history=[None, MagicMock(parts=[MagicMock(text="bar")])]))
    tool_context = MagicMock()
    tool_context.state = {}
    result = asyncio.run(agent._delegate_task("foo", "msg", tool_context))
    assert result == "bar"

@pytest.mark.asyncio
def test_agent_skills():
    agent = DummyAgent(DummyDiscovery())
    agent.skills = {"foo": ["skill1", "skill2"]}
    skills = agent._agent_skills("foo")
    assert skills == ["skill1", "skill2"]
    with pytest.raises(ValueError):
        agent._agent_skills("bar")


def test_list_agents():
    agent = DummyAgent(DummyDiscovery())
    agent.connectors = {"foo": None, "bar": None}
    agents = agent._list_agents()
    assert agents == ["foo", "bar"]


def test_tell_time():
    agent = DummyAgent(DummyDiscovery())
    result = agent._tell_time()
    assert isinstance(result, str)
    assert len(result) > 0
