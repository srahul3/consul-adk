# =============================================================================
# agents/host_agent/orchestrator.py
# =============================================================================
# 🎯 Purpose:
# Defines the OrchestratorAgent that uses a Gemini-based LLM to interpret user
# queries and delegate them to any child A2A agent discovered at startup.
# Also defines OrchestratorTaskManager to expose this logic via JSON-RPC.
# =============================================================================

import logging  # Standard library for configurable logging
import uuid  # For generating unique identifiers (e.g., session IDs)
from abc import ABC, abstractmethod

from dotenv import load_dotenv  # Utility to load environment variables from a .env file
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

from server import task_manager
from utilities.consul_discovery import ConsulDiscoveryClient

# Load the .env file so that environment variables like GOOGLE_API_KEY
# are available to the ADK client when creating LLMs
load_dotenv()

# -----------------------------------------------------------------------------
# Google ADK / Gemini imports
# -----------------------------------------------------------------------------
from google.adk.agents.llm_agent import LlmAgent, ToolUnion

# Runner: orchestrates agent, sessions, memory, and tool invocation
from google.adk.agents.readonly_context import ReadonlyContext

# InMemoryMemoryService: optional conversation memory stored in RAM
from google.adk.artifacts import InMemoryArtifactService

# InMemorySessionService: stores session state in memory (for simple demos)
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

# InMemoryArtifactService: handles file/blob artifacts (unused here)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# ReadonlyContext: passed to system prompt function to read context
from google.adk.tools.tool_context import ToolContext

# ToolContext: passed to tool functions for state and actions
from google.genai import types

from models.agent import AgentCard, AgentSkill
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskState, TaskStatus, TextPart

# -----------------------------------------------------------------------------
# A2A server-side infrastructure
# -----------------------------------------------------------------------------
from server.task_manager import InMemoryTaskManager

# -----------------------------------------------------------------------------
# Connector to child A2A agents
# -----------------------------------------------------------------------------
from utilities.agent_connect import AgentConnector

# types.Content & types.Part: used to wrap user messages for the LLM

# InMemoryTaskManager: base class providing in-memory task storage and locking

# Data models for incoming task requests and outgoing responses

# Message: encapsulates role+parts; TaskStatus/State: status enums; TextPart: text payload

# AgentConnector: lightweight wrapper around A2AClient to call other agents


# AgentCard: metadata structure for agent discovery results

# Set up module-level logger for debug/info messages
logger = logging.getLogger(__name__)


class ConsulEnabledAIAgent(ABC):
    """
    🤖 Uses a Gemini LLM to route incoming user queries,
    calling out to any discovered child A2A agents via tools.
    """

    # Define supported MIME types for input/output
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, discovery: ConsulDiscoveryClient):
        # is orchestrator agent
        self.is_orchestrator = True

        # Initialize empty collections for agents
        self.connectors = {}
        self.cards = {}
        self.skills = {}

        # Store the discovery client for later use
        self.discovery = discovery

        # Define the tools available to the agent
        self._remote_mcp_tools = {}
        self._mcp_wrappers = {}

        # User defined tools
        self._user_defined_tools = []

        # Build the internal LLM agent with our custom tools and instructions
        self._agent = self.build_agent()

        # Static user ID for session tracking across calls
        self._user_id = "orchestrator_user"

        # Runner wires up sessions, memory, artifacts, and handles agent.run()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        # Start the Consul watcher in the background
        self.initialize()

    def _set_orchestrator(self, val: bool):
        self.is_orchestrator = val

    def _get_llm_tools(self) -> list[ToolUnion]:
        """
        Returns the list of tools available to the LLM agent,
        including built-in tools and any user-defined tools.
        """
        # Built-in tools for orchestrator functionality
        built_in_tools = [
            FunctionTool(self._list_agents),
            FunctionTool(self._agent_skills),
            FunctionTool(self._delegate_task, is_async=True),
            FunctionTool(self._tell_time),
        ]

        # Combine built-in tools with MCP wrappers and user-defined tools
        if self.is_orchestrator:
            all_tools = built_in_tools + self._user_defined_tools
        else:
            all_tools = self._user_defined_tools
        return all_tools

    def _append_user_defined_tool(self, tool: ToolUnion):
        self._user_defined_tools.append(tool)


    def get_remote_mcp_tools(self):
        """
        Returns the dictionary of remote MCP tools.
        This is useful for accessing MCP tools from outside the agent.
        """
        return self._remote_mcp_tools

    def initialize(self):
        import threading

        # Create and start a daemon thread for the watcher
        background_thread = threading.Thread(target=self.run_async_watcher, daemon=True)
        background_thread.start()
        logger.info("Started Consul watcher in background thread")

    def run_async_watcher(self):
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the watch_consul coroutine in this thread's event loop
            loop.run_until_complete(self.watch_consul())
        except Exception as e:
            logger.error(f"Error in Consul watcher thread: {e}")
        finally:
            loop.close()

    async def watch_consul(self):
        """
        Background task to watch for Consul service changes.
        This properly awaits the watchConsul coroutine.
        """
        logger.info("Starting Consul service watcher")
        await self.discovery.watch_consul(callback=self.services_changed)

    async def create_mcp_tool(self, url: str) -> MCPToolset:
        """Gets tools from the File System MCP Server."""
        toolset = MCPToolset(
            connection_params=SseServerParams(
                url=url,
            )
        )
        print("MCP Toolset created successfully.")
        return toolset

    async def services_changed(self, subagents_card: {}):
        """
        Callback for when the list of agents changes.
        Updates the connectors and skills based on new agent cards.
        """
        logger.debug(f"Service changes detected, updating connectors and skills")

        agents = subagents_card["agents"]

        # Agent card map
        cards = {card.name: card for card in agents}

        # compare with existing connectors if any need to be removed or updated or added
        existing_agents = set(self.connectors.keys())
        new_agents = {card.name for card in agents}

        agent_invalidated = False
        # Agents to be added (in new list but not in existing)
        added_agents = new_agents - existing_agents
        if added_agents:
            logger.info(f"New agents discovered: {', '.join(added_agents)}")
            agent_invalidated = True
            for card in added_agents:
                # Create a new AgentConnector for each new agent
                self.create_agent_connector(cards[card])

        # Agents to be removed (in existing but not in new list)
        removed_agents = existing_agents - new_agents
        if removed_agents:
            logger.info(f"Agents no longer available: {', '.join(removed_agents)}")
            agent_invalidated = True
            for card_name in removed_agents:
                # Remove the AgentConnector
                self.remove_agent_connector(card_name)

        # Agents that might be updated (URLs or skills changed)
        updated_agents = []
        for card in agents:
            if card.name in existing_agents:
                existing_card = self.cards.get(card.name)
                if existing_card and (
                    existing_card.url != card.url
                    or AgentSkill.compare_skill_lists(existing_card.skills, card.skills)
                ):
                    agent_invalidated = True
                    self.create_agent_connector(card)

        if updated_agents:
            logger.info(f"Agents updated: {', '.join(updated_agents)}")
            agent_invalidated = True

        # --------------------------------------------------------------
        # do the same for mcps
        mcp_servers = subagents_card["mcp_servers"]

        # MCP servers map
        mcp_servers_map = {
            mcp_server["name"]: mcp_server["url"] for mcp_server in mcp_servers
        }
        existing_mcps = set(self._remote_mcp_tools.keys())
        new_mcps = {mcp_server["name"] for mcp_server in mcp_servers}

        # MCPs to be added (in new list but not in existing)
        added_mcps = new_mcps - existing_mcps
        if added_mcps:
            logger.info(f"New MCP servers discovered: {', '.join(added_mcps)}")
            agent_invalidated = True
            for mcp_name in added_mcps:
                # Create a new MCPToolset for each new MCP server
                # connection_params = StreamableHTTPConnectionParams(url=mcp_servers_map[mcp_name])
                # mcp_toolset = MCPToolset(connection_params=connection_params)
                self._remote_mcp_tools[mcp_name] = await self.create_mcp_tool(
                    mcp_servers_map[mcp_name]
                )

        # MCPs to be removed (in existing but not in new list)
        removed_mcps = existing_mcps - new_mcps
        if removed_mcps:
            logger.info(f"MCP servers no longer available: {', '.join(removed_mcps)}")
            agent_invalidated = True
            for mcp_name in removed_mcps:
                # Remove the MCPToolset
                del self._remote_mcp_tools[mcp_name]

        # MCPs that might be updated (URLs changed)
        updated_mcps = []
        for mcp_name, mcp_url in mcp_servers_map.items():
            if mcp_name in existing_mcps:
                existing_mcp_toolset = self._remote_mcp_tools.get(mcp_name)
                # for now we only check if the URL has changed
                if (
                    existing_mcp_toolset
                    and existing_mcp_toolset._connection_params.url != mcp_url
                ):
                    # Create a new MCPToolset with the updated URL
                    self._remote_mcp_tools[mcp_name] = await self.create_mcp_tool(
                        mcp_servers_map[mcp_name]
                    )
                    updated_mcps.append(mcp_name)
                    agent_invalidated = True

        if agent_invalidated:
            self._mcp_wrappers = []  # Reset the MCP wrappers list
            # expand mcp toolset into functions
            for mcp in self._remote_mcp_tools.values():
                # if mcp is of type MCPToolset, expand its tools into functions
                if isinstance(mcp, MCPToolset):
                    tools = await mcp.get_tools()
                    for tool in tools:
                        # Add each tool as a function to the agent
                        fn = self._make_wrapper(tool)
                        self._mcp_wrappers.append(FunctionTool(fn))

            logger.info("Agent configuration changed, rebuilding agent and runner")
            # If any agents or MCPs were added/removed/updated, rebuild the agent
            # Rebuild the agent with updated tools
            self._agent = self.build_agent()
            # Update the runner with the new agent
            self._runner = Runner(
                app_name=self._agent.name,
                agent=self._agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )

    def remove_agent_connector(self, card_name: str) -> None:
        """
        Remove the AgentConnector for the specified agent name.
        This is useful for cleaning up connectors when agents are no longer available.

        Args:
            card_name: The AgentCard representing the agent to remove
        """
        if card_name in self.connectors:
            del self.connectors[card_name]
            del self.cards[card_name]
            del self.skills[card_name]
            logger.info(f"Removed agent connector for {card_name}")
        else:
            logger.warning(
                f"Attempted to remove non-existent agent connector: {card_name}"
            )

    def create_agent_connector(self, card: AgentCard) -> None:
        """
        Create a new AgentConnector for the specified agent card.
        This is used to add new agents discovered via Consul.
        Args:
            card: The AgentCard representing the agent to connect to
        """
        if card.name in self.connectors:
            logger.info(f"Agent connector for {card.name} already exists, updating it.")
        self.connectors[card.name] = AgentConnector(card.name, card.url)
        self.cards[card.name] = card
        self.skills[card.name] = card.skills

    def _make_wrapper(self, tool):  # Factory creates a wrapper for a given MCPTool
        # Define an async function that accepts a single dict of args
        async def wrapper(args: dict) -> str:
            # Call the tool's run() to execute MCP command
            return await tool.run(args)

        # Name the wrapper so ADK can refer to it by the tool's name
        wrapper.__name__ = tool.name
        # updated (13/Jun/25) also add the tool description so the agent can read it
        wrapper.__doc__ = tool.description or f"Tool wrapper for MCP tool: {tool.name}"
        return wrapper

    @abstractmethod
    def build_agent(self) -> LlmAgent:
        pass

    def getTaskManager(self) -> task_manager:
        """
        Returns an instance of ConsulTaskManager that wraps the agent's invoke logic.
        This allows the agent to be used as a task manager for A2A server requests.
        """
        return ConsulTaskManager(self)

    async def invoke(self, query: str, session_id: str) -> str:
        """
        Main entry: receives a user query + session_id,
        sets up or retrieves a session, wraps the query for the LLM,
        runs the Runner (with tools enabled), and returns the final text.
        Note - function updated 28 May 2025
        Summary of changes:
        1. Agent's invoke method is made async
        2. All async calls (get_session, create_session, run_async)
            are awaited inside invoke method
        3. task manager's on_send_task updated to await the invoke call

        Reason - get_session and create_session are async in the
        "Current" Google ADK version and were synchronous earlier
        when this lecture was recorded. This is due to a recent change
        in the Google ADK code
        https://github.com/google/adk-python/commit/1804ca39a678433293158ec066d44c30eeb8e23b

        """
        # Attempt to reuse an existing session
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        # Create new if not found
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={},
            )

        # Wrap the user query in a types.Content message
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        # 🚀 Run the agent using the Runner and collect the last event
        last_event = None
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            last_event = event

        # 🧹 Fallback: return empty string if something went wrong
        if not last_event or not last_event.content or not last_event.content.parts:
            return ""

        # 📤 Extract and join all text responses into one string
        return "\n".join([p.text for p in last_event.content.parts if p.text])

    # LLM instructions fetched from KV. This provides flexibility to tune it at runtime
    def _description(self) -> str:
        return self.discovery.get_kv_variable("description")

    def _root_instruction(self, context: ReadonlyContext) -> str:
        """
        System prompt function: returns detailed instruction text for the LLM,
        including which tools it can use and a list of child agents with detailed skills.
        """
        agent_list = "\n".join(f"- {name}" for name in self.connectors)

        # Build a detailed list of agents with their skills, descriptions, tags, instructions, and examples
        agent_skills_list = []
        for name, skills in self.skills.items():
            if not skills:
                agent_skills_list.append(f"- {name}: No specific skills defined")
                continue
            skill_details = []
            for skill in skills:
                skill_name = getattr(skill, "name", getattr(skill, "id", str(skill)))
                skill_desc = getattr(skill, "description", "No description provided.")
                skill_tags = getattr(skill, "tags", [])
                skill_instruction = getattr(skill, "instruction", None)
                skill_example = getattr(skill, "examples", None)
                tags_str = f" [tags: {', '.join(skill_tags)}]" if skill_tags else ""
                instruction_str = (
                    f"\n      Instruction: {skill_instruction}"
                    if skill_instruction
                    else ""
                )
                example_str = (
                    f"\n      Example: {skill_example}" if skill_example else ""
                )
                skill_details.append(
                    f"  • {skill_name}:{tags_str}\n    Description: {skill_desc}{instruction_str}{example_str}"
                )
            agent_skills_list.append(f"- {name}:\n" + "\n".join(skill_details))
        agent_skills = "\n".join(agent_skills_list)

        default_instr = (
            "You are an orchestrator agent that routes user queries to specialized child agents.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- If required split the user query into multiple queries curated for each agent.\n"
            "- Always break down the user query into a chain of thoughts and sub-tasks.\n"
            "- If a query requires multiple steps or skills, select and sequence multiple agents as needed.\n"
            "- For complex queries, execute tasks in sequence by invoking the right agent for each sub-task.\n"
            "- Always analyze the user query to determine the best agent(s) to handle it.\n"
            "- If unsure which agent to use, check their skills first.\n"
            "- When a task fails or cannot be completed, always provide a detailed explanation of:\n"
            "- Be specific about missing capabilities - don't just say 'I can't do that', explain exactly which additional agent, skill, or tool would be needed to complete the task successfully.\n"
            "- Respond directly only for simple greetings or clarification questions.\n\n"
        )

        kv_instr = self.discovery.get_kv_variable("instruction")
        if kv_instr and len(kv_instr) > 20:
            logger.info("Using custom instruction from KV store")
            instr = kv_instr
        else:
            instr = default_instr

        return (
            instr + "\n\n"
            "Available agents:\n" + agent_list + "\n\n"
            "Agent skills (with descriptions, tags, instructions, and examples):\n"
            + agent_skills
        )

    # Tool to list all registered child agents
    def _list_agents(self) -> list[str]:
        """
        Tool function: returns the list of child-agent names currently registered.
        Called by the LLM when it wants to discover available agents.
        """
        return list(self.connectors.keys())

    # Tool to delegate a task to a specific child agent
    async def _delegate_task(
        self, agent_name: str, message: str, tool_context: ToolContext
    ) -> str:
        """
        Tool function: forwards the `message` to the specified child agent
        (via its AgentConnector), waits for the response, and returns the
        text of the last reply.
        """
        # Validate agent_name exists
        if agent_name not in self.connectors:
            raise ValueError(f"Unknown agent: {agent_name}")
        connector = self.connectors[agent_name]

        # Ensure session_id persists across tool calls via tool_context.state
        state = tool_context.state
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        session_id = state["session_id"]

        # Delegate task asynchronously and await Task result
        child_task = await connector.send_task(message, session_id)

        # Extract text from the last history entry if available
        if child_task.history and len(child_task.history) > 1:
            return child_task.history[-1].parts[0].text
        return ""

    # Tool to get the skills of a specific agent
    def _agent_skills(self, agent_name: str) -> list:
        """
        Tool function: returns the list of skills available for the specified agent.
        Called by the LLM when it needs to determine which agent has the right capabilities
        for handling a user query.

        Args:
            agent_name: Name of the agent to get skills for

        Returns:
            List of skill objects for the specified agent
        """
        # Validate agent_name exists
        if agent_name not in self.skills:
            raise ValueError(f"Unknown agent: {agent_name}")

        return self.skills[agent_name]

    # Tool to tell the current date and time
    def _tell_time(self) -> str:
        """
        Tool function: returns the current date and time as a string.
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ConsulTaskManager(InMemoryTaskManager):
    """
    🪄 TaskManager wrapper: exposes OrchestratorAgent.invoke() over the
    A2A JSON-RPC `tasks/send` endpoint, handling in-memory storage and
    response formatting.
    """

    def __init__(self, agent: ConsulEnabledAIAgent):
        super().__init__()  # Initialize base in-memory storage
        self.agent = agent  # Store our orchestrator logic

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """
        Helper: extract the user's raw input text from the request object.
        """
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Called by the A2A server when a new task arrives:
        1. Store the incoming user message
        2. Invoke the OrchestratorAgent to get a response
        3. Append response to history, mark completed
        4. Return a SendTaskResponse with the full Task
        """
        logger.info(f"OrchestratorTaskManager received task {request.params.id}")

        # Step 1: save the initial message
        task = await self.upsert_task(request.params)

        # Step 2: run orchestration logic
        user_text = self._get_user_text(request)
        response_text = await self.agent.invoke(user_text, request.params.sessionId)

        # Step 3: wrap the LLM output into a Message
        reply = Message(role="agent", parts=[TextPart(text=response_text)])
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(reply)

        # Step 4: return structured response
        return SendTaskResponse(id=request.id, result=task)
