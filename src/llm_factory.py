"""
LLM Factory - Unified Model Provider Abstraction Layer

This module provides a unified factory for creating LLM instances across different
providers (Ollama, OpenAI, DeepSeek) while maintaining a consistent interface
for LangGraph and LangChain applications.

Design Principles:
1. Provider Agnostic: Business logic doesn't need to know which provider is used
2. Easy Switching: Change providers via configuration without code changes
3. Type Safety: Full Type Hinting for better IDE support and error detection
4. Reusability: Single LLM instance can be shared across multiple agents

Usage:
    from src.llm_factory import LLMFactory
    from langchain_core.messages import HumanMessage

    # Create LLM instance
    llm = LLMFactory.get_llm(
        provider="deepseek",
        model_name="deepseek-chat",
        api_key="sk-xxx"
    )

    # Use it
    response = llm.invoke([HumanMessage(content="Hello!")])
    print(response.content)
"""

import os
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMFactory:
    """
    Unified factory for creating LLM instances across different providers.

    This factory abstracts away the differences between various LLM providers
    and provides a consistent interface for LangChain/LangGraph applications.

    Supported Providers:
    - ollama: Local models via Ollama (e.g., Qwen, Llama, Mistral)
    - openai: OpenAI API (e.g., GPT-4, GPT-3.5)
    - deepseek: DeepSeek API (OpenAI-compatible)

    Example:
        >>> # Local Ollama model
        >>> llm = LLMFactory.get_llm(provider="ollama", model_name="qwen:7b")
        >>>
        >>> # Cloud DeepSeek model
        >>> llm = LLMFactory.get_llm(
        ...     provider="deepseek",
        ...     model_name="deepseek-chat",
        ...     api_key="sk-xxx"
        ... )
    """

    # Default configurations for each provider
    DEFAULTS = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "temperature": 0.0,
            "model_name": "qwen:7b",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.0,
            "model_name": "gpt-4o-mini",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "temperature": 0.0,
            "model_name": "deepseek-chat",
        },
    }

    @staticmethod
    def get_llm(
        provider: Literal["ollama", "openai", "deepseek"] = "ollama",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> BaseChatModel:
        """
        Create an LLM instance for the specified provider.

        This method automatically selects the appropriate LangChain chat model
        class and configures it with the provided parameters.

        Args:
            provider: The LLM provider to use
                - "ollama": Local models via Ollama
                - "openai": OpenAI API
                - "deepseek": DeepSeek API (OpenAI-compatible)
            model_name: The specific model to use
                - Ollama: "qwen:7b", "llama3:8b", "mistral:7b", etc.
                - OpenAI: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", etc.
                - DeepSeek: "deepseek-chat", "deepseek-coder"
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            base_url: API base URL (overrides default for provider)
                - Ollama: Default is http://localhost:11434
                - OpenAI: Default is https://api.openai.com/v1
                - DeepSeek: Default is https://api.deepseek.com
            api_key: API key for cloud providers (required for OpenAI/DeepSeek)
            **kwargs: Additional provider-specific parameters

        Returns:
            BaseChatModel: Configured LLM instance ready for use

        Raises:
            ValueError: If provider is unknown or required parameters are missing

        Example:
            >>> # Ollama (local)
            >>> llm = LLMFactory.get_llm(provider="ollama", model_name="qwen:7b")
            >>>
            >>> # DeepSeek (cloud)
            >>> llm = LLMFactory.get_llm(
            ...     provider="deepseek",
            ...     model_name="deepseek-chat",
            ...     api_key=os.getenv("DEEPSEEK_API_KEY")
            ... )
            >>>
            >>> # Use with LangGraph
            >>> graph = create_graph(llm=llm)
        """
        # Validate provider
        if provider not in LLMFactory.DEFAULTS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: {list(LLMFactory.DEFAULTS.keys())}"
            )

        # Get default configuration
        defaults = LLMFactory.DEFAULTS[provider]

        # Use provided values or fall back to defaults
        model_name = model_name or defaults["model_name"]
        base_url = base_url or defaults["base_url"]

        # Route to appropriate provider implementation
        if provider == "ollama":
            return LLMFactory._create_ollama_llm(
                model_name=model_name,
                temperature=temperature,
                base_url=base_url,
                **kwargs,
            )
        elif provider == "openai":
            return LLMFactory._create_openai_llm(
                model_name=model_name,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key,
                **kwargs,
            )
        elif provider == "deepseek":
            return LLMFactory._create_deepseek_llm(
                model_name=model_name,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key,
                **kwargs,
            )

    @staticmethod
    def _create_ollama_llm(
        model_name: str,
        temperature: float,
        base_url: str,
        **kwargs,
    ) -> ChatOllama:
        """
        Create an Ollama LLM instance for local models.

        Ollama runs locally and doesn't require an API key.
        It's commonly used for running open-source models like Qwen, Llama, etc.

        Args:
            model_name: Ollama model name (e.g., "qwen:7b", "llama3:8b")
            temperature: Sampling temperature
            base_url: Ollama server URL (default: http://localhost:11434)
            **kwargs: Additional Ollama-specific parameters

        Returns:
            ChatOllama: Configured Ollama chat model instance
        """
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            **kwargs,
        )

    @staticmethod
    def _create_openai_llm(
        model_name: str,
        temperature: float,
        base_url: str,
        api_key: Optional[str],
        **kwargs,
    ) -> ChatOpenAI:
        """
        Create an OpenAI LLM instance.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Sampling temperature
            base_url: OpenAI API base URL
            api_key: OpenAI API key (required)
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            ChatOpenAI: Configured OpenAI chat model instance

        Raises:
            ValueError: If api_key is not provided
        """
        # Check for API key in parameter or environment variable
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "API key is required for OpenAI provider. "
                "Provide it via api_key parameter or OPENAI_API_KEY environment variable."
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @staticmethod
    def _create_deepseek_llm(
        model_name: str,
        temperature: float,
        base_url: str,
        api_key: Optional[str],
        **kwargs,
    ) -> ChatOpenAI:
        """
        Create a DeepSeek LLM instance (OpenAI-compatible).

        DeepSeek uses an OpenAI-compatible API, so we use ChatOpenAI class
        with a custom base_url.

        Args:
            model_name: DeepSeek model name (e.g., "deepseek-chat", "deepseek-coder")
            temperature: Sampling temperature
            base_url: DeepSeek API base URL
            api_key: DeepSeek API key (required)
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            ChatOpenAI: Configured DeepSeek chat model instance

        Raises:
            ValueError: If api_key is not provided
        """
        # Check for API key in parameter or environment variable
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError(
                "API key is required for DeepSeek provider. "
                "Provide it via api_key parameter or DEEPSEEK_API_KEY environment variable."
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @staticmethod
    def from_settings(settings: dict) -> BaseChatModel:
        """
        Create an LLM instance from a settings dictionary.

        This is a convenience method for creating LLM instances from
        configuration files (e.g., llm_settings.json).

        Args:
            settings: Dictionary with LLM configuration
                Expected keys:
                - provider: str ("ollama", "openai", "deepseek")
                - model_name: str (optional)
                - base_url: str (optional)
                - api_key: str (optional for Ollama, required for others)
                - temperature: float (optional, default 0.0)

        Returns:
            BaseChatModel: Configured LLM instance

        Example:
            >>> settings = {
            ...     "provider": "deepseek",
            ...     "model_name": "deepseek-chat",
            ...     "api_key": "sk-xxx"
            ... }
            >>> llm = LLMFactory.from_settings(settings)
        """
        provider = settings.get("provider", "ollama")

        return LLMFactory.get_llm(
            provider=provider,
            model_name=settings.get("model_name"),
            temperature=settings.get("temperature", 0.0),
            base_url=settings.get("base_url"),
            api_key=settings.get("api_key"),
        )

    @staticmethod
    def get_available_providers() -> list[str]:
        """
        Get list of supported LLM providers.

        Returns:
            List of provider names: ["ollama", "openai", "deepseek"]
        """
        return list(LLMFactory.DEFAULTS.keys())


# ============================================================================
# Convenience Functions
# ============================================================================


def create_ollama_llm(
    model_name: str = "qwen3:8b",
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> ChatOllama:
    """
    Convenience function for creating an Ollama LLM instance.

    Args:
        model_name: Ollama model name
        temperature: Sampling temperature
        base_url: Ollama server URL
        **kwargs: Additional parameters

    Returns:
        ChatOllama: Configured Ollama chat model
    """
    return LLMFactory.get_llm(
        provider="ollama",
        model_name=model_name,
        temperature=temperature,
        base_url=base_url,
        **kwargs,
    )


def create_openai_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **kwargs,
) -> ChatOpenAI:
    """
    Convenience function for creating an OpenAI LLM instance.

    Args:
        model_name: OpenAI model name
        temperature: Sampling temperature
        api_key: OpenAI API key
        **kwargs: Additional parameters

    Returns:
        ChatOpenAI: Configured OpenAI chat model
    """
    return LLMFactory.get_llm(
        provider="openai",
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def create_deepseek_llm(
    model_name: str = "deepseek-chat",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    **kwargs,
) -> ChatOpenAI:
    """
    Convenience function for creating a DeepSeek LLM instance.

    Args:
        model_name: DeepSeek model name
        temperature: Sampling temperature
        api_key: DeepSeek API key
        base_url: DeepSeek API base URL
        **kwargs: Additional parameters

    Returns:
        ChatOpenAI: Configured DeepSeek chat model
    """
    return LLMFactory.get_llm(
        provider="deepseek",
        model_name=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )
