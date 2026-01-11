# demo_agent.py
"""
Demo Script for UAV LangGraph Agent

This script demonstrates how to use the LangGraph ReAct agent for UAV control.
It shows the complete message flow: User -> AI (ToolCall) -> Tool (Result) -> AI (Final Answer)

Usage:
    # With default settings from llm_settings.json
    python demo_agent.py

    # With custom settings
    python demo_agent.py --provider openai-compatible --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key sk-xxx
"""
import argparse
import os
import sys

# --- 新增代码开始 ---
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
# override=True 表示如果你在系统里也设了同名变量，优先用 .env 里的
load_dotenv(override=True)
# --- 新增代码结束 ---

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import create_uav_agent_graph, create_uav_agent_graph_from_settings, print_message_history


def main():
    parser = argparse.ArgumentParser(description="UAV LangGraph Agent Demo")
    parser.add_argument(
        "--provider",
        type=str,
        help="LLM provider (e.g., openai-compatible, ollama)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="LLM API base URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model name"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="LLM API key"
    )
    parser.add_argument(
        "--uav-base-url",
        type=str,
        default="http://localhost:8000",
        help="UAV API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--use-settings",
        action="store_true",
        help="Load LLM settings from llm_settings.json"
    )

    args = parser.parse_args()

    # Create the agent graph
    if args.use_settings:
        print("="*60)
        print("Creating agent graph from llm_settings.json...")
        print("="*60)
        graph = create_uav_agent_graph_from_settings(uav_base_url=args.uav_base_url)
    elif args.provider and args.base_url:
        print("="*60)
        print(f"Creating agent graph with provider: {args.provider}")
        print(f"Base URL: {args.base_url}")
        print(f"Model: {args.model}")
        print("="*60)
        graph = create_uav_agent_graph(
            llm_provider=args.provider,
            llm_base_url=args.base_url,
            llm_model=args.model or "gpt-4o-mini",
            llm_api_key=args.api_key or os.getenv("LLM_API_KEY", ""),
            uav_base_url=args.uav_base_url,
        )
    else:
        print("="*60)
        print("Creating agent graph from llm_settings.json (default)...")
        print("="*60)
        print("\nTip: Use --use-settings flag to explicitly load from llm_settings.json")
        print("Or provide --provider, --base-url, --model, and --api-key flags\n")
        try:
            graph = create_uav_agent_graph_from_settings(uav_base_url=args.uav_base_url)
        except Exception as e:
            print(f"Error loading settings: {e}")
            print("\nPlease either:")
            print("1. Create a llm_settings.json file")
            print("2. Use --provider, --base-url, --model, and --api-key flags")
            return 1

    print("\nAgent graph created successfully!\n")

    # Demo commands
    demo_commands = [
        "请检查所有无人机及其状态，然后让第一架无人机向北移动 10 米。",
        "List all drones and their battery levels.",
        "Take off drone-001 to 15 meters altitude.",
        "Get the current weather conditions.",
        "What is the status of drone-002?",
    ]

    print("="*60)
    print("Available Demo Commands:")
    print("="*60)
    for i, cmd in enumerate(demo_commands, 1):
        print(f"{i}. {cmd}")
    print("="*60)

    # Interactive mode
    print("\nStarting interactive mode...")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nExiting...")
                break

            if not user_input:
                continue

            # Run the agent
            print("\n" + "="*60)
            print("Agent is thinking...")
            print("="*60 + "\n")

            from langchain_core.messages import HumanMessage

            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)]
            })

            # Print message history
            print_message_history(result)

            # Print final answer
            final_message = result["messages"][-1]
            if final_message.content:
                print("\n" + "="*60)
                print("Final Answer:")
                print("="*60)
                print(final_message.content)
                print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
