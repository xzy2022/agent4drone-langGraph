# langgraph_studio_entry.py
"""
LangGraph Studio Entry Point

This module provides a zero-argument factory function for LangGraph Studio.
It loads configuration from llm_settings.json and environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

from src.graph import create_uav_agent_graph_from_settings


def create_graph():
    """
    Factory function for LangGraph Studio.

    This function creates and returns a compiled UAV agent graph using
    settings from llm_settings.json and environment variables.

    Environment variables used:
    - LLM_API_KEY: API key for LLM provider (if not in llm_settings.json)
    - UAV_API_KEY: Optional API key for UAV server
    - UAV_BASE_URL: UAV server URL (defaults to http://localhost:8000)

    Returns:
        Compiled StateGraph ready for execution
    """
    # Get UAV server configuration from environment
    uav_base_url = os.getenv("UAV_BASE_URL", "http://localhost:8000")
    uav_api_key = os.getenv("UAV_API_KEY", None)
    agent_api_key = "agent_secret_key_change_in_production"

    # Create the graph using settings file
    return create_uav_agent_graph_from_settings(
        settings_path="llm_settings.json",
        uav_base_url=uav_base_url,
        uav_api_key=agent_api_key,
    )


# Export for LangGraph Studio
__all__ = ["create_graph"]
