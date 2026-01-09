# 工具状态 (已测试)

来源：在 Windows 上运行的 `tests/test_api_client.py`（参见提供的控制台输出）。

## 可用 (通过)
• `test_connection` (通过 `get_current_session` 测试)

• `list_drones`

• `get_drone_status`

• `get_task_progress`

• `get_weather`

• `get_nearby_entities`

• `take_off`

• `hover`

• `move_towards`

• `change_altitude`

• `rotate`

• `take_photo`

• `set_home`

• `return_home`

• `send_message`

• `broadcast`

• `land`


## 可用但返回了命令错误状态
这些调用返回了 HTTP 成功，但在响应中的命令 `status` 为 `error`：
• `move_to`

• `move_along_path`

• `calibrate`

• `charge`




## 因权限原因不可用 (403)
这些端点需要比当前 `agent` 角色更高的权限角色：
• `get_targets`

• `get_obstacles`


## 不可用或缺失的端点 (404)
这些端点返回了 `404 Not Found`：
• `get_waypoints` ("Target not found" / "目标未找到")

• `check_point_collision`

• `check_path_collision`



## 备注
• 仅当设置 `RUN_COMMAND_TESTS=1` 时，才会运行主动命令测试。(powershell:`$env:RUN_COMMAND_TESTS = "1"`)

• 如果您认为 `get_waypoints` 或碰撞检测端点应该存在，请确认服务器版本和 API 路由。

# 疑点 ???
可用但返回了命令错误状态， 不知道是否会由重要影响。
分明存在waypoints，可能是感知半径的影响