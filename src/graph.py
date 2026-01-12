# src\graph.py
"""
UAV Control Agent - LangGraph ReAct Loop Implementation

This module implements a basic ReAct (Reason+Act) loop using LangGraph for UAV control.
The agent can automatically understand instructions, call tools, and process return values.

Architecture:
    User Input -> Agent Node (LLM) -> Tool Node (Tool Execution) -> Agent Node -> ...
    The loop continues until the agent completes the task.

Usage:
    from src.graph import create_uav_agent_graph, UAVAPIClient

    # Create the agent graph
    graph = create_uav_agent_graph(
        llm_provider="openai-compatible",
        llm_base_url="https://api.openai.com/v1",
        llm_model="gpt-4o-mini",
        llm_api_key="sk-xxx",
        uav_base_url="http://localhost:8000"
    )

    # Run the agent
    result = graph.invoke({"messages": [("user", "List all drones and their status")]})
"""

import json
import os
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.llm_factory import LLMFactory
from src.uav_api_client import UAVAPIClient
from src.uav_tools import get_uav_tools


# ============================================================================
# State Definition
# ============================================================================


class AgentState(TypedDict):
    """
    State for the UAV Control Agent.

    The state maintains a list of messages that accumulate through the
    reasoning-action-observation loop. Each message can be from the user,
    the AI (with tool calls), or tool execution results.

    Attributes:
        messages: A list of messages with the add_messages reducer function
                  that automatically handles appending new messages.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================================
# Agent Node
# ============================================================================


def _should_continue(state: AgentState) -> Literal["tools"] | str:
    """
    Determine the next step based on the last message's tool_calls.

    This function is used in conditional edges to decide:
    - If the last message has tool_calls -> go to "tools" node
    - Otherwise -> go to END (terminate)

    Args:
        state: The current agent state containing messages

    Returns:
        "tools" if tool_calls exist, END otherwise
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # If the last message has tool_calls, continue to tools node
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end the conversation
    return END


def _create_model_node(
    llm_provider: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    tools: list[BaseTool],
):
    """
    Create a model node function for the agent.

    The model node uses an LLM with tools bound to it. When the LLM decides
    to use tools, it generates tool_calls in the response, which triggers
    the next step in the workflow.

    This function uses the LLMFactory to create the appropriate LLM instance
    based on the provider (ollama, openai, deepseek).

    Args:
        llm_provider: The LLM provider type ("ollama", "openai", "deepseek")
        llm_base_url: The base URL for the LLM API (for openai-compatible providers)
        llm_model: The model name to use
        llm_api_key: The API key for authentication (required for cloud providers)
        tools: List of LangChain BaseTool instances

    Returns:
        A callable function that can be used as a node in the graph
    """
    # Map provider names to LLMFactory provider names
    provider_mapping = {
        "ollama": "ollama",
        "openai": "openai",
        "openai-compatible": "openai",  # Map openai-compatible to openai
        "deepseek": "deepseek",
    }

    factory_provider = provider_mapping.get(llm_provider, "ollama")

    # Initialize the LLM using LLMFactory
    llm = LLMFactory.get_llm(
        provider=factory_provider,
        model_name=llm_model,
        temperature=0.0,  # Deterministic for tool use
        base_url=llm_base_url or None,
        api_key=llm_api_key if factory_provider in ["openai", "deepseek"] else None,
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    def model_node(state: AgentState) -> dict:
        """
        Agent node that processes messages and decides actions.

        This node:
        1. Receives the current state (list of messages)
        2. Invokes the LLM with tools
        3. Returns the AI response (which may include tool_calls)

        Args:
            state: Current agent state with messages

        Returns:
            Dict with the AI message added to the messages list
        """
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return model_node


# ============================================================================
# Graph Creation
# ============================================================================


def create_uav_agent_graph(
    llm_provider: str = "openai-compatible",
    llm_base_url: str = "https://api.openai.com/v1",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
    uav_base_url: str = "http://localhost:8000",
    uav_api_key: str | None = None,
) -> StateGraph:
    """
    Create a LangGraph ReAct agent for UAV control.

    This function builds a complete ReAct loop with:
    - Agent node (LLM with tool binding)
    - Tools node (automatic tool execution)
    - Conditional edges (decides whether to continue or end)

    Args:
        llm_provider: LLM provider type (default: "openai-compatible")
        llm_base_url: Base URL for the LLM API
        llm_model: Model name to use
        llm_api_key: API key for LLM authentication
        uav_base_url: Base URL for the UAV API server
        uav_api_key: Optional API key for UAV server authentication

    Returns:
        A compiled StateGraph ready for invocation

    Example:
        >>> graph = create_uav_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "List all drones and their status")]
        ... })
        >>> print(result["messages"][-1].content)
    """
    # Initialize UAV API Client
    uav_client = UAVAPIClient(base_url=uav_base_url, api_key=uav_api_key)

    # Get UAV tools
    tools = get_uav_tools(uav_client)

    # Create the model node (agent)
    model_node = _create_model_node(
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        tools=tools,
    )

    # Create the tool node for automatic tool execution
    tool_node = ToolNode(tools)

    # Initialize the state graph
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("agent", model_node)
    workflow.add_node("tools", tool_node)

    # Set the entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent to tools or END
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # Add edge from tools back to agent (so LLM can see tool results)
    workflow.add_edge("tools", "agent")

    # Compile the graph
    app = workflow.compile()

    return app


# ============================================================================
# Helper Functions
# ============================================================================


def load_llm_settings(settings_path: str = "llm_settings.json") -> dict:
    """
    Load LLM settings from a JSON configuration file.

    Args:
        settings_path: Path to the settings JSON file

    Returns:
        Dict with LLM configuration (provider, base_url, model, api_key)

    Raises:
        FileNotFoundError: If settings file doesn't exist
        KeyError: If required settings are missing
    """
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    selected_provider = settings.get("selected_provider")
    if not selected_provider:
        raise KeyError("No selected_provider in settings")

    provider_config = settings.get("provider_configs", {}).get(selected_provider)
    if not provider_config:
        raise KeyError(f"No provider config for {selected_provider}")

    return {
        "provider_name": selected_provider,
        "llm_provider": provider_config["type"],
        "llm_base_url": provider_config["base_url"],
        "llm_model": provider_config.get("default_model", "gpt-4o-mini"),
        "llm_api_key": provider_config.get("api_key", ""),
    }


def create_uav_agent_graph_from_settings(
    settings_path: str = "llm_settings.json",
    uav_base_url: str = "http://localhost:8000",
    uav_api_key: str | None = None,
) -> StateGraph:
    """
    Create a UAV agent graph using settings from a configuration file.

    This is a convenience function that loads LLM settings from a JSON file
    and creates the agent graph with those settings.

    Args:
        settings_path: Path to the LLM settings JSON file
        uav_base_url: Base URL for the UAV API server
        uav_api_key: Optional API key for UAV server authentication

    Returns:
        A compiled StateGraph ready for invocation

    Example:
        >>> graph = create_uav_agent_graph_from_settings()
        >>> result = graph.invoke({
        ...     "messages": [("user", "List all drones")]
        ... })
    """
    llm_settings = load_llm_settings(settings_path)

    # Check for API key in environment variable if not in settings
    if not llm_settings["llm_api_key"]:
        selected_provider = llm_settings.get("provider_name")
        # Try specific keys first, then generic LLM_API_KEY
        if selected_provider == "DeepSeek":
            llm_settings["llm_api_key"] = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY", "")
        elif selected_provider == "OpenAI":
            llm_settings["llm_api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
        else:
            llm_settings["llm_api_key"] = os.getenv("LLM_API_KEY", "")

    return create_uav_agent_graph(
        llm_provider=llm_settings["llm_provider"],
        llm_base_url=llm_settings["llm_base_url"],
        llm_model=llm_settings["llm_model"],
        llm_api_key=llm_settings["llm_api_key"],
        uav_base_url=uav_base_url,
        uav_api_key=uav_api_key,
    )


def print_message_history(state: AgentState) -> None:
    """
    Pretty print the message history from an agent state.

    This is useful for debugging and understanding the agent's reasoning process.

    Args:
        state: The agent state containing messages

    Example:
        >>> result = graph.invoke({"messages": [("user", "List drones")]})
        >>> print_message_history(result)
    """
    messages = state["messages"]

    for i, msg in enumerate(messages, 1):
        msg_type = msg.__class__.__name__

        print(f"\n{'='*60}")
        print(f"Message {i}: {msg_type}")
        print(f"{'='*60}")

        # Print message content
        if hasattr(msg, 'content') and msg.content:
            print(f"Content: {msg.content}")

        # Print tool calls if present
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"\nTool Calls:")
            for tool_call in msg.tool_calls:
                print(f"  - Tool: {tool_call['name']}")
                print(f"    Args: {tool_call['args']}")

        # Print name if it's a tool message
        if hasattr(msg, 'name') and msg.name:
            print(f"Tool Name: {msg.name}")

    print(f"\n{'='*60}")
    print("End of message history")
    print(f"{'='*60}\n")
