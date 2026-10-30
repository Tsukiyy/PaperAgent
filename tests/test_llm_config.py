from paper_agent.config import Settings


def test_llm_configuration_accepts_arbitrary_compatible_provider() -> None:
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="CUSTOM-GATEWAY",
        LLM_API_KEY="test-key",
        LLM_BASE_URL="https://llm.example.com/v1",
        LLM_MODEL="research-model",
        LLM_MAX_TOKENS=2048,
    )

    assert settings.llm_provider == "custom-gateway"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_base_url == "https://llm.example.com/v1"
    assert settings.llm_model == "research-model"
    assert settings.llm_max_tokens == 2048
    assert settings.llm_configured is True
    assert settings.missing_llm_settings == ()


def test_llm_configuration_reports_all_missing_connection_fields() -> None:
    settings = Settings(
        _env_file=None,
        LLM_API_KEY=None,
        LLM_BASE_URL=None,
        LLM_MODEL=None,
    )

    assert settings.llm_configured is False
    assert settings.missing_llm_settings == (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
    )
