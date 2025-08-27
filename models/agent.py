# =============================================================================
# models/agent.py
# =============================================================================
# Purpose:
# This file defines the agent-related data models used throughout the Agent2Agent (A2A) system.
# These include:
# - Capabilities that an agent supports (e.g., streaming, push notifications)
# - Metadata about the agent itself (AgentCard)
# - Metadata for each skill the agent can perform (AgentSkill)
#
# These classes help describe the agent's identity, its features, and how it interacts
# with other agents or clients in a structured and consistent way.
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# List type hint for declaring list fields
from typing import List

# BaseModel is a powerful base class from Pydantic.
# It automatically validates and converts input data into Python types
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# AgentCapabilities
# -----------------------------------------------------------------------------
# This class defines what features or protocols the agent supports.
# These capabilities can be used by A2A clients or directories to understand how to interact with the agent.
class AgentCapabilities(BaseModel):
    # Indicates if the agent can send intermediate task results through streaming
    streaming: bool = False

    # Indicates if the agent can push updates via HTTP push/webhooks
    pushNotifications: bool = False

    # If enabled, the agent keeps track of the history of task state transitions (e.g., "started", "completed")
    # Useful for debugging or auditing
    stateTransitionHistory: bool = False


# -----------------------------------------------------------------------------
# AgentSkill
# -----------------------------------------------------------------------------
# This class defines metadata about a single skill that the agent offers.
# Each skill corresponds to a specific type of task the agent can perform.
class AgentSkill(BaseModel):
    # Unique identifier for the skill (e.g., "get_time")
    id: str

    # Human-readable name for the skill (e.g., "Get Current Time")
    name: str

    # Optional description to help users understand what the skill does
    description: str | None = None

    # Optional tags to help categorize or search for the skill (e.g., ["time", "clock"])
    tags: List[str] | None = None

    # Optional example phrases that this skill might respond to
    # Useful for interfaces that suggest user queries
    examples: List[str] | None = None

    # Optional list of supported input modes (e.g., ["text", "json"])
    inputModes: List[str] | None = None

    # Optional list of supported output modes (e.g., ["text", "image"])
    outputModes: List[str] | None = None

    @staticmethod
    def compare_skill_lists(
        skills1: List["AgentSkill"], skills2: List["AgentSkill"]
    ) -> bool:
        """
        Compare two lists of AgentSkill objects for equality.

        Args:
            skills1: First list of AgentSkill objects
            skills2: Second list of AgentSkill objects

        Returns:
            bool: True if the skill lists are equivalent, False otherwise
        """
        # Handle None values
        if skills1 is None and skills2 is None:
            return True
        if skills1 is None or skills2 is None:
            return False

        # Check if the lists have different lengths
        if len(skills1) != len(skills2):
            return False

        # Create dictionaries of skills by ID for easier comparison
        skills1_dict = {skill.id: skill for skill in skills1}
        skills2_dict = {skill.id: skill for skill in skills2}

        # Check if they have the same set of skill IDs
        if skills1_dict.keys() != skills2_dict.keys():
            return False

        # Compare each skill with the same ID
        for skill_id in skills1_dict:
            skill1 = skills1_dict[skill_id]
            skill2 = skills2_dict[skill_id]

            # Compare essential attributes
            if skill1.name != skill2.name or skill1.description != skill2.description:
                return False

            # Compare tags (order independent)
            tags1 = set(skill1.tags or [])
            tags2 = set(skill2.tags or [])
            if tags1 != tags2:
                return False

            # Compare examples (order matters)
            examples1 = skill1.examples or []
            examples2 = skill2.examples or []
            if examples1 != examples2:
                return False

            # Compare input/output modes (order independent)
            input_modes1 = set(skill1.inputModes or [])
            input_modes2 = set(skill2.inputModes or [])
            if input_modes1 != input_modes2:
                return False

            output_modes1 = set(skill1.outputModes or [])
            output_modes2 = set(skill2.outputModes or [])
            if output_modes1 != output_modes2:
                return False

        # All skills are equivalent
        return True


# -----------------------------------------------------------------------------
# AgentCard
# -----------------------------------------------------------------------------
# This class provides core metadata about an agent.
# This information can be shared with a directory service or other agents
# to describe what the agent does, where to reach it, and what capabilities it supports.
class AgentCard(BaseModel):
    # Unique identifier for the agent, can be auto-generated
    id: str = None

    # Human-readable name of the agent (e.g., "Time Teller")
    name: str

    # Description of the agent's purpose or use case
    description: str

    # URL where the agent is hosted (can be used to send requests to it)
    url: str

    # Semantic version of the agent (e.g., "1.0.0")
    version: str

    # The capabilities this agent supports (uses the AgentCapabilities model above)
    capabilities: AgentCapabilities

    # List of skills (as strings) this agent can perform
    # These are references to the full AgentSkill definitions, which might be fetched elsewhere
    skills: List[AgentSkill]
