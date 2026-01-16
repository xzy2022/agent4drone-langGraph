```markdown
# Agent4Drone - LangGraph

一个基于 LangGraph 和 LangChain 构建的智能 UAV（无人机）控制代理系统。该项目支持使用多种 LLM 提供商（OpenAI, DeepSeek, Ollama）进行无人机的自然语言控制。

## 项目结构

• `demo_agent.py`: 使用 `LangGraph` 和 `src.graph` 的现代代理实现。

• `src/`: 核心逻辑和工具。
  • `graph.py`: LangGraph 状态图定义。
  • `uav_tools.py`: 无人机控制工具（起飞、移动、扫描等）。
  • `uav_api_client.py`: 用于与无人机模拟器/硬件通信的 HTTP 客户端。
  • `navigation/`: 路径规划和栅格地图逻辑（A* 算法）。

• `llm_settings.json`: LLM 提供商的配置。


## 安装

本项目需要 Python >= 3.11。

1. 克隆仓库（如果尚未完成）。
2. 创建并激活虚拟环境（推荐anaconda/miniconda）：
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 配置

1. 环境变量：
   在根目录下创建一个 `.env` 文件来存储你的 API 密钥：
   ```ini
   OPENAI_API_KEY=sk-...
   DEEPSEEK_API_KEY=sk-...
   LLM_API_KEY=sk-...
   UAV_API_KEY=...
   ```

2. LLM 设置：
   编辑 `llm_settings.json` 以配置你首选的 LLM 提供商（Ollama, OpenAI, DeepSeek 等）和模型。

## 使用方法

### 运行现代代理 (LangGraph)
推荐用于新开发。

```bash
python demo_agent.py
```
选项：
• `--use-settings`: 从 `llm_settings.json` 加载配置（默认）。
• `--provider [openai|ollama]`: 通过 CLI 覆盖提供商设置。

### 可视化开发 (LangGraph Studio)
本项目支持使用 LangGraph Studio 进行可视化开发和调试。

1. 确保已安装 `langgraph-cli` (已包含在 `requirements.txt` 中)。

2. 运行开发服务器：
   ```bash
   langgraph dev
   ```

3. 浏览器将自动打开 LangGraph Studio 界面。你可以：
   - 可视化查看代理的状态图 (`graph.py`)。
   - 交互式地发送消息并观察代理的思考过程和工具调用。
   - 修改代码后实时热重载。

配置入口文件：`langgraph_studio_entry.py`
配置文件：`langgraph.json`

### 运行测试
```bash
pytest
```
```