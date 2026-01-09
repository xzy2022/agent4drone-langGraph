"""
Test Suite for UAV Tools Factory

This test suite verifies the UAV tools factory functionality including:
- Factory function returns correct tools
- All expected tools are present in the list
- Tools properly receive client dependency injection
- Tool invoke() correctly passes parameters to client
- Pydantic schemas validate input correctly

Prerequisites:
    - No server required (uses mock client)
    - pytest installed

Usage:
    pytest tests/test_tools.py -v
    # or
    python tests/test_tools.py
"""

import json
from unittest.mock import Mock

import pytest

from langchain.tools import BaseTool

from src.uav_api_client import UAVAPIClient
from src.uav_tools import (
    create_uav_tools,
    get_uav_tools,
    ListDronesTool,
    GetDroneStatusTool,
    MoveToTool,
    TakeOffTool,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client():
    """Create a mock UAVAPIClient for testing."""
    client = Mock(spec=UAVAPIClient)
    client.base_url = "http://test.example.com"
    return client


@pytest.fixture
def tools(mock_client):
    """Create tool list using factory function."""
    return create_uav_tools(mock_client)


# Expected tools that should be returned by the factory
EXPECTED_TOOL_NAMES = [
    # Information Gathering Tools (No Parameters)
    "list_drones",
    "get_current_session",
    "get_task_progress",
    "get_weather",
    # Single-Parameter Tools
    "get_drone_status",
    "get_nearby_entities",
    "land",
    "return_home",
    "set_home",
    "calibrate",
    "take_photo",
    # Two-Parameter Tools
    "take_off",
    "change_altitude",
    "rotate",
    "hover",
    "charge",
    "broadcast",
    # Three or More Parameter Tools
    "move_to",
    "move_towards",
    "send_message",
]


# ============================================================================
# Test: Factory Function Existence
# ============================================================================


def test_create_uav_tools_is_callable():
    """Test that create_uav_tools is callable."""
    assert callable(create_uav_tools)


def test_get_uav_tools_is_callable():
    """Test that get_uav_tools is callable."""
    assert callable(get_uav_tools)


def test_get_uav_tools_is_alias():
    """Test that get_uav_tools is an alias for create_uav_tools."""
    assert create_uav_tools is get_uav_tools


# ============================================================================
# Test: Factory Returns Tool List
# ============================================================================


def test_factory_returns_list(mock_client):
    """Test that factory returns a list."""
    tools = create_uav_tools(mock_client)
    assert isinstance(tools, list)


def test_all_tools_are_basetool_instances(tools):
    """Test that all items in the list are BaseTool instances."""
    for tool in tools:
        assert isinstance(tool, BaseTool)


def test_all_expected_tools_present(tools):
    """Test that all expected tool names are present."""
    actual_names = {tool.name for tool in tools}
    for expected_name in EXPECTED_TOOL_NAMES:
        assert expected_name in actual_names, f"Expected tool '{expected_name}' not found"


def test_tool_count(tools):
    """Test that the expected number of tools are returned."""
    # Currently 20 tools (MoveAlongPathTool is commented out)
    assert len(tools) == 20


# ============================================================================
# Test: Dependency Injection
# ============================================================================


def test_all_tools_have_client_attribute(tools, mock_client):
    """Test that all tools have the client attribute properly injected."""
    for tool in tools:
        assert hasattr(tool, 'client')
        assert tool.client is mock_client


# ============================================================================
# Test: Tool Invocation - No Parameters
# ============================================================================


def test_list_drones_tool_invocation(tools):
    """Test list_drones tool invocation with no parameters."""
    # Setup mock return value
    mock_client = tools[0].client
    mock_client.list_drones.return_value = [
        {"id": "drone-001", "name": "Drone 1", "status": "idle", "battery_level": 100.0}
    ]

    # Get the tool
    list_drones_tool = next(t for t in tools if t.name == "list_drones")

    # Invoke
    result = list_drones_tool._run()

    # Verify client method was called
    mock_client.list_drones.assert_called_once()

    # Verify result is JSON string
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "drone-001"


def test_get_weather_tool_invocation(tools):
    """Test get_weather tool invocation."""
    mock_client = tools[0].client
    mock_client.get_weather.return_value = {
        "wind_speed": 5.2,
        "visibility": 10.0,
        "conditions": "clear"
    }

    weather_tool = next(t for t in tools if t.name == "get_weather")
    result = weather_tool._run()

    mock_client.get_weather.assert_called_once()

    parsed = json.loads(result)
    assert parsed["wind_speed"] == 5.2


# ============================================================================
# Test: Tool Invocation - Single Parameter
# ============================================================================


def test_get_drone_status_tool_invocation(tools):
    """Test get_drone_status tool invocation with single parameter."""
    mock_client = tools[0].client
    mock_client.get_drone_status.return_value = {
        "id": "drone-001",
        "status": "flying",
        "battery_level": 85.0,
        "position": {"x": 10.0, "y": 20.0, "z": 15.0}
    }

    status_tool = next(t for t in tools if t.name == "get_drone_status")
    result = status_tool._run(drone_id="drone-001")

    # Verify client was called with correct parameters
    mock_client.get_drone_status.assert_called_once_with(drone_id="drone-001")

    # Verify result
    parsed = json.loads(result)
    assert parsed["id"] == "drone-001"
    assert parsed["battery_level"] == 85.0


@pytest.mark.parametrize("tool_name,drone_id,method_name", [
    ("land", "drone-001", "land"),
    ("return_home", "drone-002", "return_home"),
    ("set_home", "drone-003", "set_home"),
    ("calibrate", "drone-004", "calibrate"),
    ("take_photo", "drone-005", "take_photo"),
])
def test_single_parameter_tools(tools, tool_name, drone_id, method_name):
    """Test various single-parameter tools using parametrization."""
    mock_client = tools[0].client
    mock_method = getattr(mock_client, method_name)
    mock_method.return_value = {"status": "success"}

    tool = next(t for t in tools if t.name == tool_name)
    result = tool._run(drone_id=drone_id)

    mock_method.assert_called_once_with(drone_id=drone_id)

    parsed = json.loads(result)
    assert parsed["status"] == "success"


# ============================================================================
# Test: Tool Invocation - Multiple Parameters
# ============================================================================


def test_move_to_tool_invocation(tools):
    """Test move_to tool with multiple parameters."""
    mock_client = tools[0].client
    mock_client.move_to.return_value = {"status": "success", "message": "Moving to target"}

    move_tool = next(t for t in tools if t.name == "move_to")
    result = move_tool._run(drone_id="drone-001", x=100.0, y=50.0, z=20.0)

    mock_client.move_to.assert_called_once_with(
        drone_id="drone-001",
        x=100.0,
        y=50.0,
        z=20.0
    )

    parsed = json.loads(result)
    assert parsed["status"] == "success"


def test_take_off_tool_with_explicit_altitude(tools):
    """Test take_off tool with explicit altitude parameter."""
    mock_client = tools[0].client
    mock_client.take_off.return_value = {"status": "success", "altitude": 15.0}

    takeoff_tool = next(t for t in tools if t.name == "take_off")
    result = takeoff_tool._run(drone_id="drone-001", altitude=15.0)

    mock_client.take_off.assert_called_once_with(drone_id="drone-001", altitude=15.0)

    parsed = json.loads(result)
    assert parsed["altitude"] == 15.0


def test_take_off_tool_with_default_altitude(tools):
    """Test take_off tool with default altitude."""
    mock_client = tools[0].client
    mock_client.take_off.return_value = {"status": "success", "altitude": 10.0}

    takeoff_tool = next(t for t in tools if t.name == "take_off")
    result = takeoff_tool._run(drone_id="drone-001")

    # Verify default altitude (10.0) is used
    call_args = mock_client.take_off.call_args
    assert call_args[1]['altitude'] == 10.0


def test_change_altitude_tool(tools):
    """Test change_altitude tool."""
    mock_client = tools[0].client
    mock_client.change_altitude.return_value = {"status": "success", "new_altitude": 25.0}

    tool = next(t for t in tools if t.name == "change_altitude")
    result = tool._run(drone_id="drone-001", altitude=25.0)

    mock_client.change_altitude.assert_called_once_with(drone_id="drone-001", altitude=25.0)


def test_rotate_tool(tools):
    """Test rotate tool."""
    mock_client = tools[0].client
    mock_client.rotate.return_value = {"status": "success", "heading": 90.0}

    tool = next(t for t in tools if t.name == "rotate")
    result = tool._run(drone_id="drone-001", heading=90.0)

    mock_client.rotate.assert_called_once_with(drone_id="drone-001", heading=90.0)


def test_hover_tool_with_duration(tools):
    """Test hover tool with duration parameter."""
    mock_client = tools[0].client
    mock_client.hover.return_value = {"status": "hovering"}

    tool = next(t for t in tools if t.name == "hover")
    result = tool._run(drone_id="drone-001", duration=5.0)

    mock_client.hover.assert_called_once_with(drone_id="drone-001", duration=5.0)


def test_hover_tool_without_duration(tools):
    """Test hover tool without duration (should hover indefinitely)."""
    mock_client = tools[0].client
    mock_client.hover.return_value = {"status": "hovering"}

    tool = next(t for t in tools if t.name == "hover")
    result = tool._run(drone_id="drone-001")

    # Verify duration is None
    call_args = mock_client.hover.call_args
    assert call_args[1]['duration'] is None


# ============================================================================
# Test: Communication Tools
# ============================================================================


def test_send_message_tool(tools):
    """Test send_message tool with three parameters."""
    mock_client = tools[0].client
    mock_client.send_message.return_value = {"status": "sent"}

    tool = next(t for t in tools if t.name == "send_message")
    result = tool._run(
        drone_id="drone-001",
        target_drone_id="drone-002",
        message="Hello"
    )

    mock_client.send_message.assert_called_once_with(
        drone_id="drone-001",
        target_drone_id="drone-002",
        message="Hello"
    )


def test_broadcast_tool(tools):
    """Test broadcast tool."""
    mock_client = tools[0].client
    mock_client.broadcast.return_value = {"status": "broadcasted"}

    tool = next(t for t in tools if t.name == "broadcast")
    result = tool._run(drone_id="drone-001", message="Alert")

    mock_client.broadcast.assert_called_once_with(
        drone_id="drone-001",
        message="Alert"
    )


# ============================================================================
# Test: Error Handling
# ============================================================================


def test_tool_handles_client_error_gracefully(tools):
    """Test that tools handle client errors gracefully and return error messages."""
    mock_client = tools[0].client
    mock_client.get_drone_status.side_effect = Exception("Drone not found")

    status_tool = next(t for t in tools if t.name == "get_drone_status")
    result = status_tool._run(drone_id="invalid-drone")

    # Tool should catch exception and return error message
    assert "Error" in result


# ============================================================================
# Test: Tool Metadata
# ============================================================================


def test_all_tools_have_required_metadata(tools):
    """Test that all tools have name and description attributes."""
    for tool in tools:
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert len(tool.name) > 0
        assert len(tool.description) > 0


def test_tools_with_args_schema_have_valid_schema(tools):
    """Test that tools with args_schema have valid Pydantic schemas."""
    tools_with_schema = [t for t in tools if hasattr(t, 'args_schema') and t.args_schema]

    for tool in tools_with_schema:
        schema = tool.args_schema
        assert hasattr(schema, 'model_fields')
        # Ensure schema has at least one field
        assert len(schema.model_fields) > 0


def test_tool_descriptions_are_meaningful(tools):
    """Test that tool descriptions provide useful information."""
    for tool in tools:
        # Description should be substantial (more than 20 chars)
        assert len(tool.description) > 20, f"Tool {tool.name} has too short description"


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Allow running with: python tests/test_tools.py
    pytest.main([__file__, "-v"])
