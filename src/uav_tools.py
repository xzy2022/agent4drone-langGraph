"""
UAV Control Tools - LangChain BaseTool Implementation

This module provides LangChain BaseTool implementations for the UAV Control System.
Each tool uses Pydantic schemas for type-safe parameter handling without manual JSON parsing.

Usage:
    from uav_api_client import UAVAPIClient
    from src.uav_tools import create_uav_tools

    client = UAVAPIClient(base_url="http://localhost:8000")
    tools = create_uav_tools(client)
    # tools can be used directly with LangChain agents
"""

import json
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.uav_api_client import UAVAPIClient


# ============================================================================
# Pydantic Schemas for Tool Arguments
# ============================================================================

def DroneIdField():
    return Field(
        ..., 
        description="The unique identifier of the drone (e.g., '04d6cfe7'). MUST exactly match the ID string returned by the 'list_drones' tool."
    )

class DroneIdSchema(BaseModel):
    """Schema for tools that only require a drone_id parameter."""

    drone_id: str = DroneIdField()


class TakeOffSchema(BaseModel):
    """Schema for take_off command."""
    drone_id: str = DroneIdField()
    altitude: float = Field(
        default=10.0,
        description="Target absolute altitude in meters (m).",
        ge=2.0,   # 物理硬限制：低于2米可能还在地效区
        le=500.0   # 物理硬限制：防止模型输入 9999 米
    )


class ChangeAltitudeSchema(BaseModel):
    """Schema for change_altitude command (Relative Movement)."""
    drone_id: str = DroneIdField()
    altitude: float = Field(
        ...,
        description="Altitude CHANGE (Delta) in meters (m). Positive (+) to ascend, Negative (-) to descend.",
        ge=-50.0, # 防止一次性下降太快
        le=50.0   # 防止一次性上升太快
    )

class RotateSchema(BaseModel):
    """Schema for rotate command."""
    drone_id: str = DroneIdField()
    heading: float = Field(
        ...,
        description="Target absolute heading in degrees (°). 0=North, 90=East, 180=South, 270=West.",
        ge=0.0,
        le=360.0
    )


class MoveToSchema(BaseModel):
    """Schema for move_to command (Absolute Movement)."""
    drone_id: str = DroneIdField()
    x: float = Field(..., description="Target global X coordinate in meters (m). ABSOLUTE position.")
    y: float = Field(..., description="Target global Y coordinate in meters (m). ABSOLUTE position.")
    z: float = Field(
        ..., 
        description="Target global Z altitude in meters (m). MUST be > 0.",
        ge=1.0  # 防止地下航行
    )

class MoveTowardsSchema(BaseModel):
    """Schema for move_towards command (Relative Movement)."""
    drone_id: str = DroneIdField()
    distance: float = Field(
        ..., 
        description="Distance to fly forward in meters (m).",
        gt=0.0,  # 距离必须为正数
        le=100.0 # 限制单次最大移动距离，防止飞丢
    )
    heading: Optional[float] = Field(
        default=None,
        description="Direction in degrees (°). 0=North, 90=East, 180=South, 270=West. If None, uses current drone heading.",
        ge=0.0,
        le=360.0
    )
    dz: Optional[float] = Field(
        default=None,
        description="Simultaneous vertical change in meters (m). Positive=Up, Negative=Down.",
        ge=-10.0,
        le=10.0
    )


class HoverSchema(BaseModel):
    """Schema for hover command."""
    drone_id: str = DroneIdField()
    duration: Optional[float] = Field(
        default=None,
        description="Duration to hover in seconds (s). If None, hovers until next command.",
        gt=0.0  # 时间必须是正数
    )


class ChargeSchema(BaseModel):
    """Schema for charge command."""
    drone_id: str = DroneIdField()
    charge_amount: float = Field(
        ...,
        description="Target battery percentage to reach (0-100%). Drone Must be near a waypoint with charging capability",
        ge=10.0,  # 没必要充到 5%
        le=100.0
    )


class SendMessageSchema(BaseModel):
    """Schema for send_message command."""
    drone_id: str = DroneIdField()
    target_drone_id: str = Field(
        ...,
        description="The unique ID of the RECIPIENT drone (e.g., '04d6cfe7'). MUST be different from drone_id."
    )
    message: str = Field(
        ...,
        description="The content text to send.",
        min_length=1,   # 防止发送空消息
        max_length=200  # 防止模型生成冗长的小说
    )

class BroadcastSchema(BaseModel):
    """Schema for broadcast command."""
    drone_id: str = DroneIdField()
    message: str = Field(
        ...,
        description="The content text to broadcast to ALL other drones.",
        min_length=1,
        max_length=200
    )

class MoveAlongPathSchema(BaseModel):
    """Schema for move_along_path command."""
    
    drone_id: str = DroneIdField()
    
    path_points: List[Dict[str, float]] = Field(
        ...,
        description=(
            "A sequential list of 3D coordinates (Trajectory Nodes) for the drone to follow. "
            "Structure: [{'x': 10.5, 'y': 20.0, 'z': 5.0}, ...]. "
            "Units are in meters (m). "
            "NOTE: Do NOT use this for charging stations."
        ),
        min_items=1,  # 强制至少有一个点
        max_items=50  # 防止路径过长导致超时
    )


class SessionIdSchema(BaseModel):
    """Schema for operations that accept an optional session_id."""
    
    session_id: str = Field(
        default="current",
        description="Session context ID. Defaults to 'current' active mission.",
    )


# ============================================================================
# Base Tool Classes
# ============================================================================


class UAVBaseTool(BaseTool):
    """Base class for all UAV tools with common client injection."""
    
    # 这里通常不需要 Pydantic 校验，因为 client 是注入的依赖，不是 LLM 生成的
    client: Any = Field(description="The UAV API client instance", exclude=True)

    def _run(self, **kwargs) -> str:
        """Execute the tool command and return formatted result."""
        try:
            # 增加一个通用的参数清洗，防止 LLM 幻觉出额外的参数
            valid_kwargs = {k: v for k, v in kwargs.items() if k in self.args}
            result = self._execute(**valid_kwargs)
            
            # 针对 8B 模型优化：如果返回是 dict，确保格式整洁
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, ensure_ascii=False)
            return str(result)
            
        except Exception as e:
            # 简化的错误信息，避免 Log 泄露过多内部结构给 LLM
            return f"Tool Execution Error: {str(e)}"

    def _execute(self, **kwargs) -> Any:
        """Override this method in subclasses to implement specific tool logic."""
        raise NotImplementedError("Subclasses must implement _execute method")

# ============================================================================
# Information Gathering Tools (No Parameters)
# ============================================================================


class ListDronesTool(UAVBaseTool):
    """List all available drones in the current session."""

    name: str = "list_drones"
    description: str = (
        "List all available drones in the current session with their status, battery level, and position. "
        "Use this to see what drones are available before trying to control them. "
        "No input required."
    )

    def _execute(self) -> Any:
        return self.client.list_drones()


class GetCurrentSessionTool(UAVBaseTool):
    """Get current session information."""

    name: str = "get_current_session"
    description: str = (
        "Get current session information including task type, statistics, and status. "
        "Use this to understand what mission you need to complete. "
        "No input required."
    )

    def _execute(self) -> Any:
        return self.client.get_current_session()


class GetTaskProgressTool(UAVBaseTool):
    """Get mission task progress."""

    name: str = "get_task_progress"
    description: str = (
        "Get mission task progress including completion percentage, completed targets, and status message. "
        "Use this to track mission completion and see how close you are to finishing. "
        "No input required."
    )

    def _execute(self, session_id: str = "current") -> Any:
        return self.client.get_task_progress(session_id=session_id)


class GetWeatherTool(UAVBaseTool):
    """Get current weather conditions."""

    name: str = "get_weather"
    description: str = (
        "Get current weather conditions including wind speed, visibility, and weather type. "
        "Check this before takeoff to ensure safe flying conditions. "
        "No input required."
    )

    def _execute(self) -> Any:
        return self.client.get_weather()


# ============================================================================
# Single-Parameter Tools (drone_id only)
# ============================================================================


class GetDroneStatusTool(UAVBaseTool):
    """Get detailed status of a specific drone."""

    name: str = "get_drone_status"
    description: str = (
        "Get detailed status of a specific drone including position (x, y, z), battery level, "
        "heading (0-360 degrees), current state (idle, flying, landing, etc.), and visited targets. "
        "Use this to check a drone's condition before issuing commands."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.get_drone_status(drone_id=drone_id)


class GetNearbyEntitiesTool(UAVBaseTool):
    """Get entities near a specific drone."""

    name: str = "get_nearby_entities"
    description: str = (
        "Get drones, targets, and obstacles near a specific drone within its perception radius. "
        "Returns nearby entities organized by category with their positions and distances. "
        "Use this to understand the local environment around a drone."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.get_nearby_entities(drone_id=drone_id)


class LandTool(UAVBaseTool):
    """Command a drone to land at its current position."""

    name: str = "land"
    description: str = (
        "Command a drone to land at its current position. "
        "The drone will descend vertically to the ground. "
        "The drone must be in flying state to land. "
        "Use this when a mission is complete or battery is low."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.land(drone_id=drone_id)


class ReturnHomeTool(UAVBaseTool):
    """Command a drone to return to its home position."""

    name: str = "return_home"
    description: str = (
        "Command a drone to automatically return to its home position and land. "
        "The home position is set when the drone takes off or can be manually set. "
        "Use this to recall a drone when the mission is complete or in emergencies."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.return_home(drone_id=drone_id)


class SetHomeTool(UAVBaseTool):
    """Set drone's current position as its new home position."""

    name: str = "set_home"
    description: str = (
        "Set the drone's current position as its new home position. "
        "This is useful for updating the return-to location during a mission. "
        "The drone should be at a safe location before setting home."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.set_home(drone_id=drone_id)


class CalibrateTool(UAVBaseTool):
    """Calibrate drone sensors."""

    name: str = "calibrate"
    description: str = (
        "Calibrate the drone's sensors including compass, accelerometer, and gyroscope. "
        "Calibration improves accuracy and should be done if the drone is behaving erratically. "
        "The drone should be on a level surface and stationary during calibration."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.calibrate(drone_id=drone_id)


class TakePhotoTool(UAVBaseTool):
    """Take a photo with drone camera."""

    name: str = "take_photo"
    description: str = (
        "Command a drone to take a photo with its camera. "
        "Returns photo metadata including URL, timestamp, and file size. "
        "The drone must be in flying state and have a camera available."
    )
    args_schema: type[BaseModel] = DroneIdSchema

    def _execute(self, drone_id: str) -> Any:
        return self.client.take_photo(drone_id=drone_id)


# ============================================================================
# Two-Parameter Tools
# ============================================================================


class TakeOffTool(UAVBaseTool):
    """Command a drone to take off to a specified altitude."""

    name: str = "take_off"
    description: str = (
        "Command a drone to take off from the ground to a specified altitude. "
        "The drone must be in idle or ready state on the ground. "
        "Once airborne, the drone can accept movement commands. "
        "Typical altitudes are 10-30 meters for safe operations."
    )
    args_schema: type[BaseModel] = TakeOffSchema

    def _execute(self, drone_id: str, altitude: float = 10.0) -> Any:
        return self.client.take_off(drone_id=drone_id, altitude=altitude)


class ChangeAltitudeTool(UAVBaseTool):
    """Change drone altitude while maintaining X/Y position."""

    name: str = "change_altitude"
    description: str = (
        "Change a drone's altitude while maintaining its current X/Y position. "
        "Use this to ascend or descend vertically without moving horizontally. "
        "The drone must be in flying state. "
        "Altitude can be increased (positive) or decreased (negative)."
    )
    args_schema: type[BaseModel] = ChangeAltitudeSchema

    def _execute(self, drone_id: str, altitude: float) -> Any:
        return self.client.change_altitude(drone_id=drone_id, altitude=altitude)


class RotateTool(UAVBaseTool):
    """Rotate drone to face a specific direction."""

    name: str = "rotate"
    description: str = (
        "Rotate a drone to face a specific direction without changing position. "
        "Heading: 0=North, 90=East, 180=South, 270=West. Range is 0-360 degrees. "
        "Use this to orient the drone before movement or photo capture."
    )
    args_schema: type[BaseModel] = RotateSchema

    def _execute(self, drone_id: str, heading: float) -> Any:
        return self.client.rotate(drone_id=drone_id, heading=heading)


class HoverTool(UAVBaseTool):
    """Command a drone to hover at its current position."""

    name: str = "hover"
    description: str = (
        "Command a drone to hover at its current position. "
        "If duration is provided, the drone will hover for that many seconds. "
        "If duration is not provided, the drone will hover indefinitely until the next command. "
        "Use this to wait for conditions or coordinate with other drones."
    )
    args_schema: type[BaseModel] = HoverSchema

    def _execute(self, drone_id: str, duration: Optional[float] = None) -> Any:
        return self.client.hover(drone_id=drone_id, duration=duration)


class ChargeTool(UAVBaseTool):
    """Command a drone to charge its battery."""

    name: str = "charge"
    description: str = (
        "Command a drone to charge its battery. "
        "The drone must be landed at a charging station. "
        "Specify the amount to charge in percentage points (0-100). "
        "Use this when battery is low to extend mission duration."
    )
    args_schema: type[BaseModel] = ChargeSchema

    def _execute(self, drone_id: str, charge_amount: float) -> Any:
        return self.client.charge(drone_id=drone_id, charge_amount=charge_amount)


class BroadcastTool(UAVBaseTool):
    """Broadcast a message from one drone to all other drones."""

    name: str = "broadcast"
    description: str = (
        "Broadcast a message from one drone to all other drones in the session. "
        "Use this for swarm coordination, alerts, or status updates to all drones. "
        "All drones except the sender will receive the message."
    )
    args_schema: type[BaseModel] = BroadcastSchema

    def _execute(self, drone_id: str, message: str) -> Any:
        return self.client.broadcast(drone_id=drone_id, message=message)


# ============================================================================
# Three or More Parameter Tools
# ============================================================================


class MoveToTool(UAVBaseTool):
    """Move drone to specific 3D coordinates."""

    name: str = "move_to"
    description: str = (
        "Move a drone to specific 3D coordinates (x, y, z) in the global coordinate system. "
        "The drone will fly directly to the target position. "
        "Always check for collisions first using collision checking tools if obstacles are present. "
        "The drone must be in flying state. "
        "Coordinates are in meters relative to the origin."
    )
    args_schema: type[BaseModel] = MoveToSchema

    def _execute(self, drone_id: str, x: float, y: float, z: float) -> Any:
        return self.client.move_to(drone_id=drone_id, x=x, y=y, z=z)


class MoveTowardsTool(UAVBaseTool):
    """Move drone a specific distance in a direction."""

    name: str = "move_towards"
    description: str = (
        "Move a drone a specific distance in a direction. "
        "This is relative movement from the current position. "
        "If heading is not provided, uses the drone's current heading. "
        "Optionally include vertical change (dz) for 3D movement. "
        "Use this for tactical movement without specifying absolute coordinates."
    )
    args_schema: type[BaseModel] = MoveTowardsSchema

    def _execute(self, drone_id: str, distance: float, heading: Optional[float] = None, dz: Optional[float] = None) -> Any:
        return self.client.move_towards(drone_id=drone_id, distance=distance, heading=heading, dz=dz)


class SendMessageTool(UAVBaseTool):
    """Send a message from one drone to another."""

    name: str = "send_message"
    description: str = (
        "Send a direct message from one drone to another specific drone. "
        "Use this for peer-to-peer communication in drone swarms. "
        "Only the target drone will receive the message. "
        "Use broadcast for messages to all drones."
    )
    args_schema: type[BaseModel] = SendMessageSchema

    def _execute(self, drone_id: str, target_drone_id: str, message: str) -> Any:
        return self.client.send_message(drone_id=drone_id, target_drone_id=target_drone_id, message=message)


class MoveAlongPathTool(UAVBaseTool):
    """Move drone along a path of waypoints."""

    name: str = "move_along_path"
    description: str = (
        "Move a drone along a predefined path of waypoints. "
        "The drone will visit each waypoint in sequence. "
        "Each waypoint is a dict with 'x', 'y', 'z' keys in meters. "
        "Use this for automated patrol routes or survey missions. "
        "Always check for path collisions before executing."
    )
    args_schema: type[BaseModel] = MoveAlongPathSchema

    def _execute(self, drone_id: str, waypoints: List[Dict[str, float]]) -> Any:
        return self.client.move_along_path(drone_id=drone_id, waypoints=waypoints)


# ============================================================================
# Factory Function
# ============================================================================


def create_uav_tools(client: UAVAPIClient) -> List[BaseTool]:
    """
    Create all UAV control tools for LangChain agent.

    This factory function instantiates all UAV tool classes with the provided client,
    implementing dependency injection for clean separation of concerns.

    Args:
        client: An instance of UAVAPIClient to communicate with the UAV API

    Returns:
        List of LangChain BaseTool instances ready for agent use

    Example:
        >>> from uav_api_client import UAVAPIClient
        >>> from src.uav_tools import create_uav_tools
        >>> client = UAVAPIClient(base_url="http://localhost:8000")
        >>> tools = create_uav_tools(client)
        >>> # Use tools with LangChain agent
    """
    return [
        # Information Gathering Tools (No Parameters)
        ListDronesTool(client=client),
        GetCurrentSessionTool(client=client),
        GetTaskProgressTool(client=client),
        GetWeatherTool(client=client),
        # Single-Parameter Tools
        GetDroneStatusTool(client=client),
        GetNearbyEntitiesTool(client=client),
        LandTool(client=client),
        ReturnHomeTool(client=client),
        SetHomeTool(client=client),
        CalibrateTool(client=client),
        TakePhotoTool(client=client),
        # Two-Parameter Tools
        TakeOffTool(client=client),
        ChangeAltitudeTool(client=client),
        RotateTool(client=client),
        HoverTool(client=client),
        ChargeTool(client=client),
        BroadcastTool(client=client),
        # Three or More Parameter Tools
        MoveToTool(client=client),
        MoveTowardsTool(client=client),
        SendMessageTool(client=client),
        # MoveAlongPathTool(client=client),  # 这个是原代码中整个被注释掉的，暂时存疑
    ]


# Alias for cleaner API - both names point to the same factory function
get_uav_tools = create_uav_tools
