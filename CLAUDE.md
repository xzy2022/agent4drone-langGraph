# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **UAV (Drone) Control Agent** that uses LangChain's ReAct agent pattern to enable natural language control of drone swarms through a REST API. The agent interprets user commands and uses a set of LangChain tools to interact with the UAV Control System API.

### Architecture

```
User Command → ReAct Agent → LangChain Tools → UAV API Client → UAV Server
                (LLM-powered)   (@tool decorators)    (HTTP requests)
```

**Core Components:**
- `uav_agent.py` - `UAVControlAgent` class: Main agent orchestrator using LangChain ReAct pattern
- `uav_api_client.py` - `UAVAPIClient` class: HTTP wrapper for the UAV Control System API
- `uav_langchain_tools.py` - LangChain tool definitions using `@tool` decorator pattern
- `main.py` - Tkinter GUI for interactive agent control (with optional voice input)
- `template/agent_prompt.py` - System prompt template defining agent behavior and tool usage

## Running the Application

### GUI Mode (Default)
```bash
python main.py
```
The GUI will auto-initialize and prompt for LLM provider selection if needed.

### CLI Interactive Mode
```bash
python uav_agent.py
```
Runs agent in terminal with interactive prompt selection for LLM provider.

### CLI Single Command Mode
```bash
python uav_agent.py --command "Take off drone-001 to 15 meters"
```

### Configuration

LLM providers are configured in `llm_settings.json`. The application supports:
- **Ollama** (local, no API key needed)
- **OpenAI** (requires API key)
- **OpenAI-Compatible** APIs like DeepSeek, PPIO (requires API key and base URL)

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API key (if not in llm_settings.json)
- `LLM_API_KEY` - Generic LLM API key
- `UAV_API_KEY` - UAV server authentication key (optional, defaults to USER role)

## Development Commands

### Testing the Agent
```bash
# Start the agent in debug mode
python uav_agent.py --debug --no-prompt

# Test with a specific provider
python uav_agent.py --llm-provider openai-compatible --llm-base-url https://api.example.com --llm-model my-model --llm-api-key sk-xxx
```

### UAV Server Connection
The UAV API server typically runs at `http://localhost:8000`. Ensure the server is running before starting the agent.

## Code Architecture

### Agent Flow (uav_agent.py)
1. **Initialization**: Creates `UAVAPIClient`, initializes LLM (Ollama/OpenAI), creates LangChain tools
2. **ReAct Agent Creation**: Uses `create_react_agent()` with LLM, tools, and custom prompt
3. **Execution**: `agent_executor.invoke()` runs the reasoning loop using LangChain's ReAct pattern

### LangChain Tools Pattern (uav_langchain_tools.py)
All tools use the `@tool` decorator and accept JSON string inputs for consistency:
- **No parameters**: `list_drones()`, `get_session_info()`
- **Single parameter**: `get_drone_status(input_json: str)` → `{"drone_id": "drone-001"}`
- **Multiple parameters**: `move_to(input_json: str)` → `{"drone_id": "...", "x": 100.0, "y": 50.0, "z": 20.0}`

The `create_uav_tools(client)` function returns all available tools for agent initialization.

### UAV API Client (uav_api_client.py)
Provides Pythonic wrappers around the UAV REST API endpoints:
- Drone operations: `take_off()`, `land()`, `move_to()`, `return_home()`
- Session management: `get_current_session()`, `get_task_progress()`
- Environment: `get_weather()`, `get_nearby_entities()`
- Safety: `check_path_collision()` (currently unused in agent tools)

### System Prompt (template/agent_prompt.py)
The ReAct prompt template enforces:
- Sequential reasoning: Thought → Action → Action Input → Observation
- JSON-formatted Action Input with proper escaping
- Safety rules (obstacle avoidance, battery monitoring)
- Task completion signals (responds "[TASK DONE]" at end)

## Important Patterns

### Adding New Tools
1. Add method to `UAVAPIClient` in `uav_api_client.py`
2. Create `@tool` function in `uav_langchain_tools.py` following JSON input pattern
3. Add tool to return list in `create_uav_tools()`
4. Tool automatically becomes available to agent via prompt template

### Error Handling
- The agent uses `handle_parsing_errors` to help LLM recover from JSON formatting mistakes
- `PARSING_ERROR_TEMPLATE` (template/parsing_error.py) provides structured feedback
- Max iterations set to 50 for complex multi-step tasks

### Threading in GUI
`main.py` uses daemon threads for non-blocking operations:
- Agent initialization: `_initialize_agent_worker()`
- Command execution: `_execute_command()`
- Voice recording: `record_voice_segment()`

All UI updates use `root.after(0, callback)` to schedule on main thread.

## Dependencies

- `langchain-classic` - ReAct agent framework
- `langchain-ollama` - Ollama LLM integration
- `langchain-openai` - OpenAI LLM integration
- `requests` - HTTP client for UAV API
- `tkinter` - GUI (included with Python)
- Optional: `speech_recognition`, `torch`, `transformers` for voice input (Whisper)
