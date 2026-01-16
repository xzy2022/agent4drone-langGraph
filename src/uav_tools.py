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
import math
import heapq
from typing import Any, Dict, List, Optional, Tuple, Set, ClassVar

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.uav_api_client import UAVAPIClient
from src.navigation import UAVNavigator, GridMapManager


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
        description="Target altitude in meters (m). Must be greater than or equal to 1.0.",
        ge=1.0,
        le=1000.0  # 设置合理的上限
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
        min_length=1,  # 强制至少有一个点
        max_length=50  # 防止路径过长导致超时
    )


class SmartNavigateSchema(BaseModel):
    """Schema for smart_navigate command (Long-running autonomous movement)."""
    drone_id: str = DroneIdField()
    x: float = Field(..., description="Target global X coordinate.")
    y: float = Field(..., description="Target global Y coordinate.")
    z: float = Field(..., description="Target global Z altitude.")


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
        "Set a drone to a specific altitude while maintaining its current X/Y position."
        "Use this for pure vertical ascent or descent without horizontal movement."
        "The drone must be in flying state."
        "Parameter is the target altitude value (z-coordinate), which must be greater than or equal to 1."
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
# Navigation Tool Implementations
# ============================================================================




class SmartNavigateTool(UAVBaseTool):
    """
    智能导航工具 v5.0 (乐观执行 + 反向采样试错策略)
    1. A* 全局规划：基于当前已知地图计算最优路径。
    2. 级联重试：优先尝试远距离直线飞行（乐观），若失败则逐步回退至安全感知的近处（谨慎）。
    3. 动态建图：通过行动后的感知反馈持续修正栅格地图，解决死锁。
    """
    name: str = "smart_navigate"
    description: str = (
        "终极导航工具。采用 A* 规划与‘乐观执行-回退重试’策略， "
        "在开阔地带极速飞行，在复杂地形稳健避障。支持死循环自动破解和动态地图学习。"
    )
    args_schema: type[BaseModel] = SmartNavigateSchema

    # 算法配置
    GRID_RESOLUTION: float = 5.0
    DEFAULT_SENSING_RADIUS: float = 30.0
    SAFE_FACTOR: float = 0.8  # 安全系数，用于确定“保底安全点”

    # --- 1. 新增：类级别的持久化存储 ---
    # 格式：{ (session_id, drone_id): GridMapManager_Instance }
    # 使用 ClassVar 确保所有工具实例共享这份数据
    _persistent_maps: ClassVar[Dict[Tuple[str, str], GridMapManager]] = {}

    def _get_map_cache_path(self, session_id: str, drone_id: str) -> str:
        """获取地图缓存文件路径"""
        import os
        cache_dir = ".map_cache"
        return os.path.join(cache_dir, f"map_{session_id}_{drone_id}.json")

    def _get_candidate_waypoints(self, full_path: List[dict], current_pos: dict, sense_radius: float) -> List[dict]:
        """
        策略优化版 v2:
        1. 终点 (尝试直连)
        2. 最近的拐点 (A* 路径的第一条直线段终点，保证不切墙角)
        3. 第一步 (保底，仅移动一格)
        """
        candidates = []
        
        # --- 1. 最乐观：终点 ---
        if full_path:
            candidates.append(full_path[-1])
        
        # --- 2. 次乐观：寻找第一个拐点 (Turning Point) ---
        # 拐点定义：路径方向发生改变的那个节点
        turning_point = None
        
        # 路径至少要有 3 个点才能判断拐弯 (起点 -> P1 -> P2)
        if len(full_path) >= 3:
            # 计算第一步的方向向量 (dx1, dy1)
            # 注意：full_path[0] 通常是当前格子，full_path[1] 是下一步
            dx1 = full_path[1]['x'] - full_path[0]['x']
            dy1 = full_path[1]['y'] - full_path[0]['y']
            
            # 遍历后续节点，寻找方向变化
            for i in range(2, len(full_path)):
                dx2 = full_path[i]['x'] - full_path[i-1]['x']
                dy2 = full_path[i]['y'] - full_path[i-1]['y']
                
                # 判断方向是否一致 (允许微小浮点误差)
                # 如果方向向量变了，说明 i-1 这个点就是拐点
                if abs(dx2 - dx1) > 0.1 or abs(dy2 - dy1) > 0.1:
                    turning_point = full_path[i-1]
                    break
            
            # 如果遍历完都没找到拐点，说明整条路径都是直的
            # 此时拐点就是终点，无需重复添加
        
        # 添加拐点 (去重)
        if turning_point:
            # 简单的去重逻辑：如果离终点太近，就没必要加了
            dist_to_final = ((turning_point['x'] - candidates[0]['x'])**2 + 
                             (turning_point['y'] - candidates[0]['y'])**2)**0.5
            if dist_to_final > 1.0: 
                candidates.append(turning_point)

        # --- 3. 保底：A* 路径的第一个有效移动节点 ---
        # full_path[0] 是起点，full_path[1] 是迈出的第一步
        if len(full_path) > 1:
            first_step = full_path[1]
            
            # 去重检查 (避免和拐点或终点重复)
            is_duplicate = False
            for c in candidates:
                if abs(c['x'] - first_step['x']) < 0.1 and abs(c['y'] - first_step['y']) < 0.1:
                    is_duplicate = True
            
            if not is_duplicate:
                candidates.append(first_step)
                
        return candidates

    def _truncate_path_to_known(self, full_path: List[dict], grid_map: GridMapManager) -> Tuple[List[dict], bool]:
        """
        悲观行动：检查路径是否进入未知区域。
        如果是，则在进入未知区域前的 1-2 个栅格处截断路径。
        返回：(截断后的路径, 是否被截断)
        """
        if not full_path:
            return [], False

        for i, wp in enumerate(full_path):
            gx, gy = grid_map._to_grid(wp['x']), grid_map._to_grid(wp['y'])
            status = grid_map.get_status(gx, gy)
            
            # 发现进入未知区域 (-2.0)
            if status == -2.0:
                # 往前回退 1-2 个点
                truncate_idx = max(0, i - 2)
                print(f"[SmartNav] 路径进入未知区域 (栅格 {gx}, {gy})，在索引 {truncate_idx} 处进行悲观截断。")
                return full_path[:truncate_idx + 1], True
                
        return full_path, False

    def _execute(self, drone_id: str, x: float, y: float, z: float) -> str:
        final_target = {"x": x, "y": y, "z": z}
        print(f"\n[SmartNav] 指挥官指令下达: 终点 ({x}, {y}, {z})")

        # --- 获取 Session ID ---
        try:
            session_info = self.client.get_current_session()
            session_id = session_info.get("id", "default_session")
        except Exception as e:
            print(f"[SmartNav] 获取会话 ID 失败: {e}，将使用默认会话。")
            session_id = "default_session"

        map_key = (session_id, drone_id)
        cache_path = self._get_map_cache_path(session_id, drone_id)

        # --- 2. 获取或创建持久化地图 ---
        if map_key not in self._persistent_maps:
            # 尝试从磁盘加载
            loaded_map = GridMapManager.load_from_disk(cache_path)
            if loaded_map:
                print(f"[SmartNav] 从磁盘恢复了无人机 {drone_id} 在会话 {session_id} 的历史地图记忆 (已探索 {len(loaded_map.obstacles)} 个障碍)")
                self._persistent_maps[map_key] = loaded_map
            else:
                print(f"[SmartNav] 为无人机 {drone_id} (会话: {session_id}) 初始化全新地图记忆...")
                self._persistent_maps[map_key] = GridMapManager(
                    resolution=self.GRID_RESOLUTION, 
                    inflation=1
                )
        else:
            print(f"[SmartNav] 命中内存缓存：加载无人机 {drone_id} 的历史地图记忆 (已探索 {len(self._persistent_maps[map_key].obstacles)} 个障碍)")
            
        # 获取引用
        grid_map = self._persistent_maps[map_key]
        
        try:
            # 1. 获取初始状态
            status = self.client.get_drone_status(drone_id)
            pos_info = status.get("position", status)
            current_pos = {"x": pos_info.get("x", 0), "y": pos_info.get("y", 0), "z": pos_info.get("z", 0)}
            sense_radius = status.get("perceived_radius", self.DEFAULT_SENSING_RADIUS)
            
            # 初始感知建图
            nearby = self.client.get_nearby_entities(drone_id)
            grid_map.add_obstacles_from_entities(nearby)
            # 标记当前位置周围为已探索
            grid_map.mark_explored_area(current_pos['x'], current_pos['y'], sense_radius)
            
            # 立即保存一次地图
            grid_map.save_to_disk(cache_path)

            # --- 阶段一：垂直高度调整 (决定巡航高度) ---
            # 逻辑：取当前高度和目标高度的较大值作为“巡航高度 (fly_z)”
            # 如果目标更高，先爬升；如果当前更高，保持高度平飞，最后再降落。
            cruise_z = max(current_pos['z'], final_target['z'])
            
            # 如果需要爬升 (且高度差大于 0.5m)
            if cruise_z > current_pos['z'] + 0.5:
                print(f"[SmartNav] 目标高度较高，正在原地爬升至巡航高度 {cruise_z}m...")
                self.client.change_altitude(drone_id, cruise_z)
                # 爬升后必须更新当前状态，否则 A* 依然会以为在低空规划
                current_pos['z'] = cruise_z 
                # 重新扫描高空环境（虽然通常高空视野更好）
                nearby = self.client.get_nearby_entities(drone_id)
                grid_map.add_obstacles_from_entities(nearby)

        except Exception as e:
            return f"初始化/起飞失败: {str(e)}"
        
        max_loops = 50
        loop_count = 0

        # --- 阶段二：水平巡航导航 ---
        while loop_count < max_loops:
            loop_count += 1
            
            # 计算 2D 平面距离 (忽略高度差，因为我们在巡航高度平飞)
            dist_2d = ((current_pos['x']-final_target['x'])**2 + (current_pos['y']-final_target['y'])**2)**0.5
            
            if dist_2d < 2.0:
                print(f"[SmartNav] 水平位置已抵达，准备最后进近。")
                break # 跳出循环，进入降落阶段

            # 构造当前的巡航目标 (X, Y 它是终点，但 Z 是巡航高度)
            cruise_target = {"x": final_target['x'], "y": final_target['y'], "z": cruise_z}

            # 1. A* 全局规划 (基于当前的 cruise_z)
            print(f"[SmartNav] 正在计算巡航航线 (高度 {current_pos['z']:.1f}m)... (Cycle: {loop_count})")
            full_path = grid_map.a_star_search(current_pos, cruise_target)
            

            # 如果 A* 返回空（因为起点终点在同一格）或者路径太短，但我们还在循环里（说明距离 > 2.0m）
            if not full_path or len(full_path) == 0:
                start_grid = (grid_map._to_grid(current_pos['x']), grid_map._to_grid(current_pos['y']))
                end_grid = (grid_map._to_grid(cruise_target['x']), grid_map._to_grid(cruise_target['y']))
                
                # 如果是在同一个格子里，或者距离非常近，强制直飞
                if start_grid == end_grid or dist_2d < self.GRID_RESOLUTION * 1.5:
                    print("[SmartNav] 目标在当前栅格内，切换为直连模式。")
                    full_path = [cruise_target] # 手动构造一条直达路径
                else:
                    return "导航终止：路径被物理遮断，无法规划 A* 路径。"
            
            # 2. 悲观截断：如果路径进入未知区域，则缩短路径
            path_to_execute, is_truncated = self._truncate_path_to_known(full_path, grid_map)
            if is_truncated:
                print(f"[SmartNav] 目标处于未知或必经未知区域，采用截断后的临时目标。")

            # 3. 生成级联候选点 (基于截断后的路径)
            candidates = self._get_candidate_waypoints(path_to_execute, current_pos, sense_radius)
            move_success = False

            # 3. 级联尝试
            for i, wp in enumerate(candidates):
                d_attempt = ((wp['x']-current_pos['x'])**2 + (wp['y']-current_pos['y'])**2)**0.5
                if i == 0:
                    desc = "直达临时目标点"
                elif i == 1:
                    desc = "前往第一拐点"
                else:  # i == 2
                    desc = "前往临近点"
                    
                print(f"[SmartNav] 方案 {i+1} ({desc}): 目标距离 {d_attempt:.1f}m | 目标点 ({wp['x']:.1f}, {wp['y']:.1f}, {cruise_z:.1f})", end="", flush=True)
                
                try:
                    # 注意：这里强制使用 cruise_z 进行平飞
                    move_result = self.client.move_to(drone_id, wp['x'], wp['y'], cruise_z)
                    
                    if isinstance(move_result, dict) and move_result.get('status') == 'error':
                        msg = move_result.get('message', '').lower()
                        if "obstacle" in msg or "collision" in msg:
                            print("[遇阻]")
                            continue 
                        else:
                            return f"执行异常: {msg}"
                    else:
                        print("[成功]")
                        move_success = True
                        break 
                except Exception as e:
                    return f"API 通讯故障: {str(e)}"

            # 4. 后验感知与动态建图
            new_status = self.client.get_drone_status(drone_id)
            pos = new_status.get("position", new_status)
            current_pos = {"x": pos.get("x", 0), "y": pos.get("y", 0), "z": pos.get("z", 0)}
            
            nearby_entities = self.client.get_nearby_entities(drone_id)
            grid_map.add_obstacles_from_entities(nearby_entities)
            # 标记新位置周围为已探索
            grid_map.mark_explored_area(current_pos['x'], current_pos['y'], sense_radius)

            # if not move_success:
            #     print("[SmartNav] 警告：水平路径受阻，尝试垂直机动...")
            #     # 死锁处理：如果在巡航高度还被堵，尝试再升高 10m
            #     cruise_z += 10.0
            #     self.client.change_altitude(drone_id, cruise_z)
            #     current_pos['z'] = cruise_z # 更新本地状态

        if loop_count >= max_loops:
            return "导航超时：执行步数过多。"

        # --- 阶段三：垂直降落/调整 (Descend Phase) ---
        # 此时已经到达目标 X/Y 上方，高度为 cruise_z
        # 如果 cruise_z 比 final_target['z'] 高，则下降
        
        if abs(current_pos['z'] - final_target['z']) > 0.5:
            print(f"[SmartNav] 正在从 {current_pos['z']:.1f}m 调整高度至 {final_target['z']:.1f}m...")
            try:
                self.client.change_altitude(drone_id, final_target['z'])
                return "成功抵达目的地 (包含高度调整)"
            except Exception as e:
                return f"最终高度调整失败: {str(e)}"
        
        return "成功抵达目的地"


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
        # MoveToTool(client=client),        # 这两个都没有必要，导航全部使用SmartNavigateTool
        # MoveTowardsTool(client=client),   # 这两个都没有必要，导航全部使用SmartNavigateTool
        SendMessageTool(client=client),
        SmartNavigateTool(client=client),
        # MoveAlongPathTool(client=client),  # 这个是原代码中整个被注释掉的，暂时存疑
    ]


# Alias for cleaner API - both names point to the same factory function
get_uav_tools = create_uav_tools
