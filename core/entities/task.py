from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from .position import Position

class TaskStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class Task:
    """
    Task/Mission Entity.
    Represents a target to visit or an action to perform.
    """
    id: str
    position: Optional[Position] = None
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assigned_drone_id: Optional[str] = None
    
    def mark_completed(self):
        self.status = TaskStatus.COMPLETED

    def mark_in_progress(self, drone_id: str):
        self.status = TaskStatus.IN_PROGRESS
        self.assigned_drone_id = drone_id
