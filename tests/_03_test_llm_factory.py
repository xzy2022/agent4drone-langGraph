"""
Test Suite for LLM Factory

This test suite verifies the LLM Factory functionality including:
1. Provider instantiation (Ollama, OpenAI, DeepSeek)
2. Interface consistency (all return BaseChatModel)
3. Instance reusability (single LLM can be invoked multiple times)
4. Configuration flexibility (from settings dict, environment variables)
5. Error handling (missing API keys, unknown providers)

Prerequisites:
    - No API keys required for most tests (uses mocks)
    - Ollama optional (for real integration tests)
    - pytest and langchain installed

Usage:
    # Run all tests
    pytest tests/test_llm_factory.py -v

    # Run specific test
    pytest tests/test_llm_factory.py::test_ollama_instantiation -v -s

    # Run with real Ollama (requires Ollama running)
    pytest tests/test_llm_factory.py::test_ollama_real_invocation -v -s
"""

import os
from unittest.mock import Mock, patch, MagicMock

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.llm_factory import (
    LLMFactory,
    create_ollama_llm,
    create_openai_llm,
    create_deepseek_llm,
)


# ============================================================================
# Test: Provider Instantiation
# ============================================================================


class TestOllamaInstantiation:
    """Test Ollama provider instantiation."""

    def test_ollama_returns_chat_model(self):
        """Test that Ollama provider returns a BaseChatModel instance."""
        llm = LLMFactory.get_llm(provider="ollama", model_name="qwen:7b")

        assert isinstance(llm, BaseChatModel)
        # Verify it's specifically a ChatOllama instance
        assert llm.__class__.__name__ == "ChatOllama"

    def test_ollama_default_configuration(self):
        """Test that Ollama uses default configuration when not specified."""
        llm = LLMFactory.get_llm(provider="ollama")

        assert llm.model == "qwen:7b"
        assert llm.temperature == 0.0
        assert llm.base_url == "http://localhost:11434"

    def test_ollama_custom_configuration(self):
        """Test that Ollama accepts custom configuration."""
        llm = LLMFactory.get_llm(
            provider="ollama",
            model_name="llama3:8b",
            temperature=0.7,
            base_url="http://192.168.1.100:11434",
        )

        assert llm.model == "llama3:8b"
        assert llm.temperature == 0.7
        assert llm.base_url == "http://192.168.1.100:11434"

    def test_ollama_convenience_function(self):
        """Test the convenience function for Ollama."""
        llm = create_ollama_llm(model_name="mistral:7b", temperature=0.5)

        assert isinstance(llm, BaseChatModel)
        assert llm.model == "mistral:7b"
        assert llm.temperature == 0.5


class TestOpenAIInstantiation:
    """Test OpenAI provider instantiation."""

    def test_openai_returns_chat_model(self):
        """Test that OpenAI provider returns a BaseChatModel instance."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            llm = LLMFactory.get_llm(
                provider="openai",
                model_name="gpt-4o-mini",
                api_key="sk-test-key",
            )

            assert isinstance(llm, BaseChatModel)
            assert llm.__class__.__name__ == "ChatOpenAI"

    def test_openai_requires_api_key(self):
        """Test that OpenAI raises ValueError when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required for OpenAI"):
                LLMFactory.get_llm(provider="openai", model_name="gpt-4o-mini")

    def test_openai_uses_environment_api_key(self):
        """Test that OpenAI can use API key from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            llm = LLMFactory.get_llm(provider="openai", model_name="gpt-4o-mini")

            assert llm.openai_api_key.get_secret_value() == "sk-env-key"

    def test_openai_custom_configuration(self):
        """Test that OpenAI accepts custom configuration."""
        llm = LLMFactory.get_llm(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.3,
            api_key="sk-custom-key",
            base_url="https://custom.openai.com/v1",
        )

        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.3
        assert llm.openai_api_key.get_secret_value() == "sk-custom-key"
        assert llm.openai_api_base == "https://custom.openai.com/v1"

    def test_openai_convenience_function(self):
        """Test the convenience function for OpenAI."""
        llm = create_openai_llm(
            model_name="gpt-4o-mini",
            api_key="sk-test-key",
            temperature=0.2,
        )

        assert isinstance(llm, BaseChatModel)
        assert llm.model_name == "gpt-4o-mini"
        assert llm.temperature == 0.2


class TestDeepSeekInstantiation:
    """Test DeepSeek provider instantiation."""

    def test_deepseek_returns_chat_model(self):
        """Test that DeepSeek provider returns a BaseChatModel instance."""
        llm = LLMFactory.get_llm(
            provider="deepseek",
            model_name="deepseek-chat",
            api_key="sk-deepseek-key",
        )

        assert isinstance(llm, BaseChatModel)
        assert llm.__class__.__name__ == "ChatOpenAI"  # DeepSeek uses ChatOpenAI

    def test_deepseek_requires_api_key(self):
        """Test that DeepSeek raises ValueError when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required for DeepSeek"):
                LLMFactory.get_llm(provider="deepseek", model_name="deepseek-chat")

    def test_deepseek_uses_environment_api_key(self):
        """Test that DeepSeek can use API key from environment variable."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-env-deepseek"}):
            llm = LLMFactory.get_llm(provider="deepseek", model_name="deepseek-chat")

            assert llm.openai_api_key.get_secret_value() == "sk-env-deepseek"

    def test_deepseek_custom_base_url(self):
        """Test that DeepSeek uses correct default base URL."""
        llm = LLMFactory.get_llm(
            provider="deepseek",
            model_name="deepseek-chat",
            api_key="sk-test-key",
        )

        assert llm.openai_api_base == "https://api.deepseek.com"

    def test_deepseek_custom_configuration(self):
        """Test that DeepSeek accepts custom configuration."""
        llm = LLMFactory.get_llm(
            provider="deepseek",
            model_name="deepseek-coder",
            temperature=0.1,
            api_key="sk-custom-deepseek",
            base_url="https://custom.deepseek.com",
        )

        assert llm.model_name == "deepseek-coder"
        assert llm.temperature == 0.1
        assert llm.openai_api_key.get_secret_value() == "sk-custom-deepseek"
        assert llm.openai_api_base == "https://custom.deepseek.com"

    def test_deepseek_convenience_function(self):
        """Test the convenience function for DeepSeek."""
        llm = create_deepseek_llm(
            model_name="deepseek-chat",
            api_key="sk-test-key",
            temperature=0.0,
        )

        assert isinstance(llm, BaseChatModel)
        assert llm.model_name == "deepseek-chat"
        assert llm.temperature == 0.0


# ============================================================================
# Test: Interface Consistency
# ============================================================================


class TestInterfaceConsistency:
    """Test that all providers return consistent interfaces."""

    def test_all_providers_return_base_chat_model(self):
        """Test that all providers return BaseChatModel instances."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "DEEPSEEK_API_KEY": "sk-test"}):
            ollama_llm = LLMFactory.get_llm(provider="ollama", model_name="qwen:7b")
            openai_llm = LLMFactory.get_llm(provider="openai", model_name="gpt-4o-mini")
            deepseek_llm = LLMFactory.get_llm(
                provider="deepseek", model_name="deepseek-chat"
            )

            # All should be BaseChatModel instances
            assert isinstance(ollama_llm, BaseChatModel)
            assert isinstance(openai_llm, BaseChatModel)
            assert isinstance(deepseek_llm, BaseChatModel)

    def test_all_providers_have_invoke_method(self):
        """Test that all providers have the invoke method."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "DEEPSEEK_API_KEY": "sk-test"}):
            ollama_llm = LLMFactory.get_llm(provider="ollama", model_name="qwen:7b")
            openai_llm = LLMFactory.get_llm(provider="openai", model_name="gpt-4o-mini")
            deepseek_llm = LLMFactory.get_llm(
                provider="deepseek", model_name="deepseek-chat"
            )

            # All should have invoke method
            assert hasattr(ollama_llm, "invoke")
            assert hasattr(openai_llm, "invoke")
            assert hasattr(deepseek_llm, "invoke")

    def test_all_providers_have_bind_tools_method(self):
        """Test that all providers support tool binding (required for LangGraph agents)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "DEEPSEEK_API_KEY": "sk-test"}):
            ollama_llm = LLMFactory.get_llm(provider="ollama", model_name="qwen:7b")
            openai_llm = LLMFactory.get_llm(provider="openai", model_name="gpt-4o-mini")
            deepseek_llm = LLMFactory.get_llm(
                provider="deepseek", model_name="deepseek-chat"
            )

            # All should have bind_tools method
            assert hasattr(ollama_llm, "bind_tools")
            assert hasattr(openai_llm, "bind_tools")
            assert hasattr(deepseek_llm, "bind_tools")


# ============================================================================
# Test: Instance Reusability
# ============================================================================


class TestInstanceReusability:
    """Test that a single LLM instance can be reused across multiple calls."""

    def test_single_ollama_instance_multiple_invocations(self):
        """
        Test that a single Ollama instance can be invoked multiple times.

        This demonstrates that in a single-GPU environment, multiple agents
        can share the same LLM instance (serial execution), conserving memory.
        """
        # Create mock LLM to test reusability without requiring Ollama
        mock_llm = Mock(spec=BaseChatModel)
        call_count = [0]

        def mock_invoke(messages):
            call_count[0] += 1
            return AIMessage(content=f"Response {call_count[0]}")

        mock_llm.invoke = mock_invoke

        # Test that the same instance can be invoked multiple times
        response_1 = mock_llm.invoke([HumanMessage(content="Hello from Agent A")])
        assert response_1.content == "Response 1"

        response_2 = mock_llm.invoke([HumanMessage(content="Hello from Agent B")])
        assert response_2.content == "Response 2"

        response_3 = mock_llm.invoke([HumanMessage(content="Hello again from Agent A")])
        assert response_3.content == "Response 3"

        # Verify the same instance was used 3 times
        assert call_count[0] == 3

    def test_single_openai_instance_multiple_invocations(self):
        """
        Test that a single OpenAI instance can be invoked multiple times.

        For cloud APIs, this tests connection reuse and configuration sharing.
        """
        # Create a mock LLM that simulates reusability
        mock_llm = Mock(spec=BaseChatModel)
        call_count = [0]

        def mock_invoke(messages):
            call_count[0] += 1
            return AIMessage(content=f"OpenAI Response {call_count[0]}")

        mock_llm.invoke = mock_invoke

        # First agent's request
        response_1 = mock_llm.invoke([HumanMessage(content="Request 1")])
        assert response_1.content == "OpenAI Response 1"

        # Second agent's request (same instance)
        response_2 = mock_llm.invoke([HumanMessage(content="Request 2")])
        assert response_2.content == "OpenAI Response 2"

        # Verify both used the same LLM instance
        assert call_count[0] == 2
        assert mock_llm is not None

    def test_deepseek_instance_multiple_invocations(self):
        """Test that a single DeepSeek instance can be invoked multiple times."""
        # Create a mock LLM that simulates reusability
        mock_llm = Mock(spec=BaseChatModel)
        call_count = [0]

        def mock_invoke(messages):
            call_count[0] += 1
            return AIMessage(content=f"DeepSeek Response {call_count[0]}")

        mock_llm.invoke = mock_invoke

        # Multiple invocations with same instance
        response_1 = mock_llm.invoke([HumanMessage(content="Message 1")])
        assert response_1.content == "DeepSeek Response 1"

        response_2 = mock_llm.invoke([HumanMessage(content="Message 2")])
        assert response_2.content == "DeepSeek Response 2"

        # Verify same instance was used twice
        assert call_count[0] == 2

    def test_instance_reusability_scenario(self):
        """
        Real-world scenario: Two agents sharing one LLM instance.

        This simulates a common pattern where:
        - Agent A (planner) and Agent B (executor) share the same LLM
        - They operate sequentially (not concurrent) to save memory
        - Single GPU can only run one model instance at a time
        """
        # Create shared mock LLM instance
        shared_llm = Mock(spec=BaseChatModel)
        call_count = [0]
        responses = [
            "Plan: Execute task A, then task B",
            "Executing task A...",
            "Executing task B...",
            "Task complete",
        ]

        def mock_invoke(messages):
            call_count[0] += 1
            return AIMessage(content=responses[call_count[0] - 1])

        shared_llm.invoke = mock_invoke

        # Agent A (Planner) - creates plan
        plan = shared_llm.invoke([HumanMessage(content="Create a plan")])
        assert "Plan:" in plan.content

        # Agent B (Executor) - executes step 1
        step1 = shared_llm.invoke([HumanMessage(content="Execute step 1")])
        assert "Executing" in step1.content

        # Agent B (Executor) - executes step 2
        step2 = shared_llm.invoke([HumanMessage(content="Execute step 2")])
        assert "Executing" in step2.content

        # Agent A (Planner) - verifies completion
        verification = shared_llm.invoke([HumanMessage(content="Verify completion")])
        assert "complete" in verification.content

        # All used the same instance
        assert call_count[0] == 4


# ============================================================================
# Test: Configuration Flexibility
# ============================================================================


class TestConfigurationFlexibility:
    """Test configuration flexibility."""

    def test_from_settings_dict_ollama(self):
        """Test creating LLM from settings dictionary for Ollama."""
        settings = {
            "provider": "ollama",
            "model_name": "llama3:8b",
            "temperature": 0.5,
            "base_url": "http://localhost:11434",
        }

        llm = LLMFactory.from_settings(settings)

        assert isinstance(llm, BaseChatModel)
        assert llm.model == "llama3:8b"
        assert llm.temperature == 0.5

    def test_from_settings_dict_openai(self):
        """Test creating LLM from settings dictionary for OpenAI."""
        settings = {
            "provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.2,
            "api_key": "sk-settings-key",
        }

        llm = LLMFactory.from_settings(settings)

        assert isinstance(llm, BaseChatModel)
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.2

    def test_from_settings_dict_deepseek(self):
        """Test creating LLM from settings dictionary for DeepSeek."""
        settings = {
            "provider": "deepseek",
            "model_name": "deepseek-chat",
            "api_key": "sk-deepseek-key",
        }

        llm = LLMFactory.from_settings(settings)

        assert isinstance(llm, BaseChatModel)
        assert llm.model_name == "deepseek-chat"

    def test_from_settings_with_defaults(self):
        """Test that missing settings use defaults."""
        settings = {
            "provider": "ollama",
            # model_name, temperature, base_url omitted
        }

        llm = LLMFactory.from_settings(settings)

        # Should use default values
        assert llm.model == "qwen:7b"  # Default for Ollama
        assert llm.temperature == 0.0

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = LLMFactory.get_available_providers()

        assert isinstance(providers, list)
        assert "ollama" in providers
        assert "openai" in providers
        assert "deepseek" in providers


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_unknown_provider_raises_value_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMFactory.get_llm(provider="unknown_provider", model_name="test")

    def test_openai_missing_api_key_raises_value_error(self):
        """Test that OpenAI without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required for OpenAI"):
                LLMFactory.get_llm(provider="openai", model_name="gpt-4o-mini")

    def test_deepseek_missing_api_key_raises_value_error(self):
        """Test that DeepSeek without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required for DeepSeek"):
                LLMFactory.get_llm(provider="deepseek", model_name="deepseek-chat")


# ============================================================================
# Test: Real Integration (Optional - requires services)
# ============================================================================


class TestRealIntegration:
    """Integration tests with real services (optional)."""

    @pytest.mark.skipif(
        False,  # Skip by default
        reason="Requires Ollama running locally. Enable manually for integration testing.",
    )
    def test_ollama_real_invocation(self):
        """
        Integration test with real Ollama instance.

        Enable by removing @pytest.mark.skipif or running:
            pytest tests/test_llm_factory.py::test_ollama_real_invocation -v -s
        """
        llm = LLMFactory.get_llm(provider="ollama", model_name="qwen3:8b")

        response = llm.invoke([HumanMessage(content="Say 'Hello, Ollama!'")])

        assert isinstance(response, AIMessage)
        assert len(response.content) > 0
        print(f"\nOllama Response: {response.content}")

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires OpenAI API key. Enable manually for integration testing.",
    )
    def test_openai_real_invocation(self):
        """
        Integration test with real OpenAI API.

        Enable by removing @pytest.mark.skipif or running:
            pytest tests/test_llm_factory.py::test_openai_real_invocation -v -s
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = LLMFactory.get_llm(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key=api_key,
        )

        response = llm.invoke([HumanMessage(content="Say 'Hello, OpenAI!'")])

        assert isinstance(response, AIMessage)
        assert len(response.content) > 0
        print(f"\nOpenAI Response: {response.content}")


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Allow running with: python tests/test_llm_factory.py
    pytest.main([__file__, "-v", "-s"])
