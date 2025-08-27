# consul-adk
This is an AI Agent Development Kit(ADK) built on top of Google ADK to implement Agentic Service Mesh. 
Using this and Consul CE/ENT, you can discover and scale the Agents and MCP servers and create a desired agent graph using Service intentions.
The ADK help to create the desired client's for the Agents and MCP servers as soon as the intentions are defined in Consul.

## Installation

```bash
pip install consul-adk
```

## Usage Example

Below is a simple example to get you started:

A simple agent with hardcoded instructions
```python
# 🧠 Gemini-based AI agent provided by Google's ADK
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
from utilities.consul_agent import ConsulEnabledAIAgent

class WeatherAgent(ConsulEnabledAIAgent):

    def build_agent(self) -> LlmAgent:
        """
        ⚙️ Creates and configures a Gemini agent with weather capabilities.

        Returns:
            LlmAgent: A configured agent object from Google's ADK
        """
        # Manually adding the WeatherTool to the agent's tools
        tools = [
            FunctionTool(WeatherTool(name="WeatherTool", description="Gets the realtime weather for one or more locations").get_weather)
        ]

        # Automatically add MCP tools as they are registered in Consul and Intentions are defined
        for tool_set in self._remote_mcp_tools.values():
            tools.append(tool_set)

        return LlmAgent(
            model="gemini-1.5-flash-latest",  # Gemini model version
            name="weather_agent",  # Name of the agent
            description="Provides weather information for multiple locations",  # Description for metadata
            instruction="You are a helpful weather assistant. When asked about weather, extract the 'locations' parameter as a list (even for a single location), then use the 'WeatherTool' to provide weather information. When multiple locations are requested, organize your response clearly with headings for each location. Be friendly and informative. If user asks for locations with specific weather conditions, first get weather for several cities in that region, then recommend ones that match the criteria.",
            # System prompt
            tools=tools,
        )
from google.adk.tools import BaseTool
class WeatherTool(BaseTool):
    """A tool that provides weather information for multiple locations.

    Attributes:
        name: The name of the tool.
        description: A brief description of what the tool does.
    """
    name = "WeatherTool"
    description = "Gets the current weather information for one or more specified locations"

    async def get_weather(self, locations: List[str]) -> List[dict[str, Any]]:
        # Hey developer, implement the logic to fetch weather data here
```

An orchestrator agent which adapts its instructions by reading the child-agent card.
```python
class OrchestratorAgent(ConsulEnabledAIAgent):

    def build_agent(self) -> LlmAgent:
        """
        Construct the Gemini-based LlmAgent with tools
        """
        tools = [
            self._list_agents, # built-in tool to list all registered agents
            self._agent_skills, # built-in tool to list all skills of a specific agent
            self._delegate_task, # built-in tool to delegate tasks to other agents
            self._tell_time # built-in tool to tell the current time
        ]
        # Automatically add MCP tools as they are registered in Consul and Intentions are defined
        for tool_set in self._remote_mcp_tools.values():
            # Add each MCPToolset to the agent's tools
            tools.append(tool_set)

        ai_agent = LlmAgent(
            model="gemini-1.5-flash-latest",
            name="orchestrator_agent",
            description="Delegates user queries to child A2A agents based on intent.",
            instruction=self._root_instruction, # dynamically generated instruction based on current state of the child agents
            tools=tools
        )

        return ai_agent

```


## Features

- Auto discover AI agents and creates their client connections in this agent as soon as Consul intentions are defined
- Auto-configure agents and MCP servers
- Send and manage tasks between agents

## Documentation

For more details, see the [internal.md](internal.md) or explore the `client/`, `models/`, and `server/` directories.


## Contributing
We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for details on how to get started.

Please run below command to ensure your code compiles before it is submitted:
```bash
python -m build
```
