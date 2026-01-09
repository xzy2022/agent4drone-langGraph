"""
Integration Test for UAV LangGraph Agent

This test suite verifies the LangGraph ReAct loop functionality:
1. Agent can understand natural language instructions
2. Agent automatically calls tools in sequence
3. ToolMessage is automatically appended to State
4. LLM performs reasoning based on tool results

Prerequisites:
    - UAV API server running (or use mock for testing)
    - LLM API configured (or use mock for testing)
    - pytest installed
    - langgraph, langchain_openai installed

Usage:
    # Test with real API (requires server running)
    #   set RUN_REAL_AGENT_TESTS=1 then:
    #   pytest tests/test_agent.py::test_react_loop_with_real_api -v -s

    # Test with mock client (no server required)
    pytest tests/test_agent.py::test_react_loop_with_mock -v -s

    # Run all tests
    python tests/test_agent.py
"""

import json
import os
from unittest.mock import Mock, patch

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import (
    create_uav_agent_graph,
    create_uav_agent_graph_from_settings,
    _should_continue,
    AgentState,
    print_message_history,
)
from src.uav_api_client import UAVAPIClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_uav_client():
    """Create a mock UAVAPIClient for testing without real server."""
    client = Mock(spec=UAVAPIClient)
    client.base_url = "http://test.example.com"

    # Mock list_drones response
    client.list_drones.return_value = [
        {
            "id": "drone-001",
            "name": "Drone 1",
            "status": "idle",
            "battery_level": 100.0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading": 0.0,
        },
        {
            "id": "drone-002",
            "name": "Drone 2",
            "status": "idle",
            "battery_level": 95.0,
            "position": {"x": 10.0, "y": 10.0, "z": 0.0},
            "heading": 90.0,
        },
    ]

    # Mock move_towards response (moving 10 meters north = heading 0)
    client.move_towards.return_value = {
        "status": "success",
        "message": "Drone moving 10 meters north",
        "drone_id": "drone-001",
    }

    return client


@pytest.fixture
def mock_llm():
    """Create a mock LLM that simulates tool calling."""
    llm = Mock()

    # Simulate LLM response that calls list_drones
    first_response = AIMessage(
        content="I'll check all drones and then move the first one north.",
        tool_calls=[
            {
                "name": "list_drones",
                "args": {},
                "id": "call_001",
                "type": "tool_call",
            }
        ],
    )

    # Simulate LLM response that calls move_towards after seeing drone list
    second_response = AIMessage(
        content="Now I'll move the first drone 10 meters north.",
        tool_calls=[
            {
                "name": "move_towards",
                "args": {
                    "drone_id": "drone-001",
                    "distance": 10.0,
                    "heading": 0.0,  # North
                },
                "id": "call_002",
                "type": "tool_call",
            }
        ],
    )

    # Simulate final response after tool execution
    final_response = AIMessage(
        content="Task completed! I've checked all drones and moved drone-001 10 meters north."
    )

    llm.invoke.side_effect = [first_response, second_response, final_response]
    return llm


# ============================================================================
# Test: Helper Functions
# ============================================================================


def test_should_continue_with_tool_calls():
    """Test that _should_continue returns 'tools' when tool_calls present."""
    state: AgentState = {
        "messages": [
            AIMessage(
                content="I'll check the drones",
                tool_calls=[
                    {
                        "name": "list_drones",
                        "args": {},
                        "id": "call_001",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }

    result = _should_continue(state)
    assert result == "tools"


def test_should_continue_without_tool_calls():
    """Test that _should_continue returns END when no tool_calls."""
    state: AgentState = {
        "messages": [
            AIMessage(content="Task completed successfully!")
        ]
    }

    result = _should_continue(state)
    assert result == "__end__"


def test_should_continue_with_empty_messages():
    """Test that _should_continue returns END with empty message list."""
    state: AgentState = {
        "messages": []
    }

    result = _should_continue(state)
    assert result == "__end__"


# ============================================================================
# Test: Graph Creation
# ============================================================================


def test_create_uav_agent_graph_basic():
    """Test that the graph can be created with basic parameters."""
    # Mock UAV client to avoid server requirement
    with patch('src.graph.UAVAPIClient') as mock_client_class:
        mock_client = Mock(spec=UAVAPIClient)
        mock_client_class.return_value = mock_client

        # Mock get_uav_tools to return empty list for simplicity
        with patch('src.graph.get_uav_tools') as mock_tools:
            mock_tools.return_value = []

            graph = create_uav_agent_graph(
                llm_provider="openai-compatible",
                llm_base_url="https://api.openai.com/v1",
                llm_model="gpt-4o-mini",
                llm_api_key="test-key",
            )

            assert graph is not None
            # Verify UAV client was initialized
            mock_client_class.assert_called_once()


def test_create_uav_agent_graph_from_settings():
    """Test that the graph can be created from settings file."""
    # Mock settings loading
    mock_settings = {
        "llm_provider": "openai-compatible",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": "test-key",
    }

    with patch('src.graph.load_llm_settings') as mock_load:
        mock_load.return_value = mock_settings

        # Mock UAV client and tools
        with patch('src.graph.UAVAPIClient') as mock_client_class:
            mock_client = Mock(spec=UAVAPIClient)
            mock_client_class.return_value = mock_client

            with patch('src.graph.get_uav_tools') as mock_tools:
                mock_tools.return_value = []

                graph = create_uav_agent_graph_from_settings()

                assert graph is not None
                mock_load.assert_called_once()


# ============================================================================
# Test: ReAct Loop with Mock (Integration Test)
# ============================================================================


def test_react_loop_with_mock():
    """
    Integration test: Verify the ReAct loop works with mocked components.

    This test simulates:
    1. User asks to check all drones and move first one north
    2. Agent calls list_drones
    3. Agent calls move_towards with correct parameters
    4. Agent provides final answer

    Key verification: ToolMessage is automatically appended to State.
    """
    # Mock UAV client
    with patch('src.graph.UAVAPIClient') as mock_client_class:
        mock_client = Mock(spec=UAVAPIClient)
        mock_client.list_drones.return_value = [
            {
                "id": "drone-001",
                "name": "Drone 1",
                "status": "idle",
                "battery_level": 100.0,
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "heading": 0.0,
            }
        ]
        mock_client.move_towards.return_value = {
            "status": "success",
            "message": "Moving drone-001 10 meters north",
        }
        mock_client_class.return_value = mock_client

        # Mock LLMFactory with simulated tool calling behavior
        with patch('src.graph.LLMFactory') as mock_llm_factory_class:
            from langchain_core.language_models.chat_models import BaseChatModel

            mock_llm = Mock(spec=BaseChatModel)

            # Create mock responses
            response1 = AIMessage(
                content="I'll check all drones first.",
                tool_calls=[
                    {
                        "name": "list_drones",
                        "args": {},
                        "id": "call_001",
                        "type": "tool_call",
                    }
                ],
            )

            response2 = AIMessage(
                content="I found the drones. Now I'll move the first one north.",
                tool_calls=[
                    {
                        "name": "move_towards",
                        "args": {"drone_id": "drone-001", "distance": 10.0, "heading": 0.0},
                        "id": "call_002",
                        "type": "tool_call",
                    }
                ],
            )

            response3 = AIMessage(
                content="Task completed! I've checked all drones and moved drone-001 10 meters north."
            )

            mock_llm.invoke.side_effect = [response1, response2, response3]
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_factory_class.get_llm.return_value = mock_llm

            # Create graph
            graph = create_uav_agent_graph(
                llm_provider="openai-compatible",
                llm_base_url="https://api.openai.com/v1",
                llm_model="gpt-4o-mini",
                llm_api_key="test-key",
            )

            # Run the agent
            user_input = "请检查所有无人机及其状态，然后让第一架无人机向北移动 10 米。"
            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)]
            })

            # Verify the result
            assert result is not None
            assert "messages" in result

            messages = result["messages"]

            # Expected message flow:
            # 1. HumanMessage (user input)
            # 2. AIMessage (first AI response with list_drones call)
            # 3. ToolMessage (result of list_drones)
            # 4. AIMessage (second AI response with move_towards call)
            # 5. ToolMessage (result of move_towards)
            # 6. AIMessage (final answer)

            assert len(messages) == 6, f"Expected 6 messages, got {len(messages)}"

            # Verify first message is user input
            assert isinstance(messages[0], HumanMessage)
            assert user_input in messages[0].content

            # Verify second message has tool_calls
            assert isinstance(messages[1], AIMessage)
            assert len(messages[1].tool_calls) == 1
            assert messages[1].tool_calls[0]["name"] == "list_drones"

            # Verify third message is ToolMessage (automatic State update)
            assert isinstance(messages[2], ToolMessage)
            assert messages[2].name == "list_drones"
            assert "drone-001" in messages[2].content

            # Verify fourth message has second tool call
            assert isinstance(messages[3], AIMessage)
            assert len(messages[3].tool_calls) == 1
            assert messages[3].tool_calls[0]["name"] == "move_towards"

            # Verify fifth message is ToolMessage (automatic State update)
            assert isinstance(messages[4], ToolMessage)
            assert messages[4].name == "move_towards"

            # Verify final message is AI response
            assert isinstance(messages[5], AIMessage)
            assert len(messages[5].tool_calls) == 0

            # Verify client methods were called
            mock_client.list_drones.assert_called_once()
            mock_client.move_towards.assert_called_once_with(
                drone_id="drone-001",
                distance=10.0,
                heading=0.0,
                dz=None,
            )


# ============================================================================
# Test: Message History Printing
# ============================================================================


def test_print_message_history(capsys):
    """Test the print_message_history helper function."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="List all drones"),
            AIMessage(
                content="I'll check the drones",
                tool_calls=[
                    {
                        "name": "list_drones",
                        "args": {},
                        "id": "call_001",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content='[{"id": "drone-001", "status": "idle"}]',
                tool_call_id="call_001",
                name="list_drones",
            ),
        ]
    }

    print_message_history(state)

    captured = capsys.readouterr()
    output = captured.out

    # Verify all messages are printed
    assert "Message 1: HumanMessage" in output
    assert "Message 2: AIMessage" in output
    assert "Message 3: ToolMessage" in output
    assert "list_drones" in output


# ============================================================================
# Test: Real API Integration (Optional - requires server)
# ============================================================================


@pytest.mark.skipif(
    os.getenv("RUN_REAL_AGENT_TESTS") not in {"1", "true", "True"},
    reason="Set RUN_REAL_AGENT_TESTS=1 to run with real UAV server and LLM API."
)
def test_react_loop_with_real_api():
    """
    Integration test with real API (requires server and API key).

    This test verifies the agent works with actual APIs.
    Enable by removing the @pytest.mark.skipif decorator or using:
        pytest tests/test_agent.py::test_react_loop_with_real_api -v -s
    """
    # Create graph from settings file
    graph = create_uav_agent_graph_from_settings()

    # Run the agent
    user_input = "请检查所有无人机及其状态，然后让第一架无人机向北移动 10 米。"
    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Verify result
    assert result is not None
    assert "messages" in result

    # Print message history for debugging
    print("\n" + "="*60)
    print("Message History:")
    print("="*60)
    print_message_history(result)

    # Verify we have multiple messages (tool calls were made)
    assert len(result["messages"]) >= 3

    # Verify final message is AI response
    final_message = result["messages"][-1]
    assert isinstance(final_message, AIMessage)


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Allow running with: python tests/test_agent.py
    pytest.main([__file__, "-v", "-s"])
