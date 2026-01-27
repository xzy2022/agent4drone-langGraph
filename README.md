# Agent4Drone - LangGraph

基于 LangGraph / LangChain 的智能 UAV（无人机）控制代理项目。支持 OpenAI、DeepSeek、Ollama 等多种 LLM 提供商，通过自然语言完成无人机查询、起飞、移动、扫描等任务，并可在 LangGraph Studio 中可视化调试。

## 项目亮点

- 采用 LangGraph ReAct 循环，消息 -> 工具 -> 反馈的完整链路清晰可追踪
- 统一的 UAV API Client + Tool 封装，便于对接真实或模拟飞行器
- 支持多 LLM 提供商与配置化切换，便于本地/云端混合使用

## 项目结构

- `demo_agent.py`：命令行演示入口，支持参数覆盖 LLM 配置
- `src/graph.py`：LangGraph 状态图与 ReAct 逻辑
- `src/uav_tools.py`：无人机控制工具（起飞、移动、扫描、查询等）
- `src/uav_api_client.py`：与 UAV API 服务通信的 HTTP 客户端
- `src/navigation/`：路径规划与栅格地图（A*）
- `llm_settings.json`：多提供商 LLM 配置
- `langgraph_studio_entry.py` / `langgraph.json`：Studio 可视化配置

## 环境要求

- Python >= 3.11

## 安装

```bash
pip install -r requirements.txt
```

## 配置

1. 环境变量（可选）：在项目根目录创建 `.env`

```ini
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
LLM_API_KEY=sk-...
UAV_API_KEY=...
```

2. LLM 配置：编辑 `llm_settings.json` 选择提供商与默认模型

## 快速开始

```bash
python demo_agent.py
```

常用参数：

- `--use-settings`：从 `llm_settings.json` 加载配置
- `--provider` / `--base-url` / `--model` / `--api-key`：命令行覆盖配置
- `--uav-base-url`：UAV API 服务地址（默认 `http://localhost:8000`）

## 可视化调试（LangGraph Studio）

```bash
langgraph dev
```

入口文件：`langgraph_studio_entry.py`  
配置文件：`langgraph.json`

## 运行测试

```bash
pytest
```
