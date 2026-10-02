import pytest

from app.config import ConfigError, load_config


def test_defaults_apply_when_only_model_is_set(valid_env: dict[str, str]) -> None:
    config = load_config(valid_env)

    assert config.llm_model == "test-model"
    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_temperature == 0.2


def test_values_are_parsed_and_normalised(valid_env: dict[str, str]) -> None:
    env = {
        **valid_env,
        "LLM_BASE_URL": "http://example.test:1234/v1/",
        "LLM_TEMPERATURE": "0.9",
        "LOG_LEVEL": "debug",
        "SHOW_TOKEN_USAGE": "false",
    }

    config = load_config(env)

    assert config.llm_base_url == "http://example.test:1234/v1"
    assert config.llm_temperature == 0.9
    assert config.log_level == "DEBUG"
    assert config.show_token_usage is False


def test_missing_model_names_the_environment_variable() -> None:
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        load_config({})


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("LLM_TEMPERATURE", "3.5"),
        ("LLM_TOP_P", "0"),
        ("LLM_BASE_URL", "localhost:11434"),
        ("MAX_HISTORY_MESSAGES", "1"),
        ("LOG_LEVEL", "VERBOSE"),
    ],
)
def test_invalid_values_are_rejected(valid_env: dict[str, str], variable: str, value: str) -> None:
    with pytest.raises(ConfigError, match=variable):
        load_config({**valid_env, variable: value})


def test_api_key_is_hidden_from_repr(valid_env: dict[str, str]) -> None:
    config = load_config({**valid_env, "LLM_API_KEY": "super-secret"})

    assert "super-secret" not in repr(config)
