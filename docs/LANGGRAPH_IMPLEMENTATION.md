# LangGraph ReAct 循环实现

## 概述

本文档描述了用于无人机控制智能体（UAV Control Agent）的基于 LangGraph 的 ReAct（推理+行动）循环实现。该智能体可以自动理解自然语言指令，按顺序调用工具，并处理返回值，而无需任何手动解析代码。

## 架构

```
用户输入 -> 智能体节点 (LLM) -> 工具节点 (执行) -> 智能体节点 -> ...
                 |                       |
                 v                       v
             工具调用                  工具消息
           (如果需要)               (自动添加到 State)
```

### 消息流

1.  **用户消息 (User Message)**：用户提供自然语言指令
2.  **AI 消息 (AI Message，包含 tool_calls)**：LLM 决定使用哪些工具
3.  **工具消息 (Tool Message)**：工具执行，结果被**自动**添加到 State（状态）中
4.  **AI 消息 (AI Message)**：LLM 处理工具结果并决定下一步行动
5.  **循环继续**：直到 LLM 决定任务完成（不再产生 tool_calls）



## 测试命令

运行测试套件：

```bash
# 运行所有测试
pytest tests/test_agent.py -v

# 运行特定测试
pytest tests/test_agent.py::test_react_loop_with_mock -v -s

# 使用真实 API 运行（需要服务器和 API 密钥）
pytest tests/test_agent.py::test_react_loop_with_real_api -v -s
```



## 测试示例

```
============================================================
Message History:
============================================================

============================================================
Message 1: HumanMessage
============================================================
Content: 请检查所有无人机及其状态，然后让第一架无人机向北移动 10 米。

============================================================
Message 2: AIMessage
============================================================

Tool Calls:
  - Tool: list_drones
    Args: {}

============================================================
Message 3: ToolMessage
============================================================
Content: [
  {
    "id": "ab176b88",
    "name": "New Drone 1",
    "model": "Model-A",
    "status": "idle",
    "position": {
      "x": 3.0,
      "y": 277.0,
      "z": 0.0
    },
    "heading": 180.0,
    "speed": 0.0,
    "battery_level": 97.81116727929688,
    "max_speed": 20.0,
    "max_altitude": 120.0,
    "battery_capacity": 4000.0,
    "perceived_radius": 100.0,
    "task_radius": 10.0,
    "home_position": {
      "x": 3.0,
      "y": 277.0,
      "z": 4.0
    },
    "created_at": 1767965610.2152936,
    "last_updated": 1767967998.8862848
  },
  {
    "id": "eda1e74a",
    "name": "New Drone 2",
    "model": "Model-A",
    "status": "idle",
    "position": {
      "x": 297.0,
      "y": 0.0,
      "z": 0.0
    },
    "heading": 0.0,
    "speed": 0.0,
    "battery_level": 100.0,
    "max_speed": 20.0,
    "max_altitude": 120.0,
    "battery_capacity": 4000.0,
    "perceived_radius": 100.0,
    "task_radius": 10.0,
    "home_position": {
      "x": 297.0,
      "y": 0.0,
      "z": 0.0
    },
    "created_at": 1767965617.7817056,
    "last_updated": 1767965617.7817056
  }
]
Tool Name: list_drones

============================================================
Message 4: AIMessage
============================================================

Tool Calls:
  - Tool: move_to
    Args: {'drone_id': 'ab176b88', 'x': 3, 'y': 287, 'z': 0}

============================================================
Message 5: ToolMessage
============================================================
Content: {
  "command_id": "5f8b93e0",
  "drone_id": "ab176b88",
  "command": "move_to",
  "status": "error",
  "message": "Cannot move from current status: DroneStatus.IDLE"
}
Tool Name: move_to

============================================================
Message 6: AIMessage
============================================================

Tool Calls:
  - Tool: take_off
    Args: {'altitude': 10, 'drone_id': 'ab176b88'}

============================================================
Message 7: ToolMessage
============================================================
Content: {
  "command_id": "c8d42a57",
  "drone_id": "ab176b88",
  "command": "take_off",
  "status": "success",
  "message": "Drone taking off to altitude 10.0m (battery: -0.5%/-20mAh, remaining: 97.3%/3892mAh)"
}
Tool Name: take_off

============================================================
Message 8: AIMessage
============================================================

Tool Calls:
  - Tool: move_to
    Args: {'drone_id': 'ab176b88', 'x': 3, 'y': 287, 'z': 10}

============================================================
Message 9: ToolMessage
============================================================
Content: {
  "command_id": "27a6ec37",
  "drone_id": "ab176b88",
  "command": "move_to",
  "status": "success",
  "message": "Drone moved to position (3.0, 287.0, 10.0), heading: 0.0° (battery: -0.2%/-10mAh, remaining: 97.1%/3883mAh)"
}
Tool Name: move_to

============================================================
Message 10: AIMessage
============================================================
Content: 第一架无人机（ID: ab176b88）已成功向北移动10米，当前位置为 (3.0, 287.0, 10.0)。移动过 程中电池消耗0.2%（剩余97.1%），飞行状态稳定。

============================================================
End of message history
============================================================

```

