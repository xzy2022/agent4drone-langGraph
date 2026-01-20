from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from .position import Position

class DroneState(Enum):
    LANDED = "LANDED"       # 未起飞（降落后进入该状态）
    HOVERING = "HOVERING"   # 悬停中（显示指定悬停命令）
    FLYING = "FLYING"       # 飞行中（正在执行飞行）
    IDLE = "IDLE"           # 空闲（飞在空中，但是没有移动，也没有指定悬停）
    UNKNOWN = "UNKNOWN"     # 未知
    CHARGING = "CHARGING"   # 充电中

@dataclass
class Drone:
    """
    Drone Entity.
    """
    id: str
    position: Position = field(default_factory=lambda: Position(0,0,0))
    battery: float = 100.0
    heading: float = 0.0
    state: DroneState = DroneState.IDLE
    
    def takeoff(self, altitude: float) -> bool:
        if self.state == DroneState.LANDED:
            self.position.z = altitude
            self.state = DroneState.IDLE
            return True
        return False

    def land(self) -> bool:
        if self.state == DroneState.IDLE:
            self.position.z = 0
            self.state = DroneState.LANDED
            return True
        return False

    # 目前感觉充电完全可以使用 update_battery 代替
    # def charge(self, can_charge: bool) -> bool:
    #     if can_charge and self.state == DroneState.LANDED:
    #         self.state = DroneState.CHARGING
    #         return True
    #     return False

    def hover(self) -> bool:
        if self.state == DroneState.IDLE or self.state == DroneState.FLYING:
            self.state = DroneState.HOVERING
            return True
        return False

    def move(self, position: Position) -> bool:
        if self.state == DroneState.IDLE or self.state == DroneState.HOVERING or self.state == DroneState.FLYING:
            self.position = position
            return True
        return False

    def rotate(self, heading: float) -> bool:
        if self.state == DroneState.IDLE or self.state == DroneState.HOVERING or self.state == DroneState.FLYING:
            self.heading = heading
            return True
        return False
    
    def update_battery(self, battery: float) -> bool:
        if battery >= 0 and battery <= 100:
            self.battery = battery
            return True
        return False


