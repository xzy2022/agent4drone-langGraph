from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

class AgentStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    WAITING = "WAITING"
    RUNNING = "RUNNING"

@dataclass
class Agent:
    """
    Agent Entity.
    Represents an intelligent agent with its context and state.
    """
    id: str
    system_prompt: str = ""
    input: Optional[Any] = None
    output: Optional[Any] = None
    status: AgentStatus = AgentStatus.WAITING
