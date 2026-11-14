
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from autonomous_data_detective.agents import investigator
from autonomous_data_detective.agents.schemas import InvestigationReport


def test_build_investigator_agent_configures_model_and_agent():
    settings = SimpleNamespace(
        openai_model="gpt-4.1-mini",
        openai_api_key=SecretStr("test-api-key"),
    )

    fake_model = MagicMock()
    fake_agent = MagicMock()

    with (
        patch.object(
            investigator,
            "get_settings",
            return_value=settings,
        ),
        patch.object(
            investigator,
            "ChatOpenAI",
            return_value=fake_model,
        ) as mock_chat_openai,
        patch.object(
            investigator,
            "create_agent",
            return_value=fake_agent,
        ) as mock_create_agent,
    ):
        result = investigator.build_investigator_agent()

    assert result is fake_agent

    mock_chat_openai.assert_called_once_with(
        model="gpt-4.1-mini",
        api_key="test-api-key",
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    agent_kwargs = mock_create_agent.call_args.kwargs

    assert agent_kwargs["model"] is fake_model
    assert agent_kwargs["name"] == "investigator_agent"
    assert agent_kwargs["response_format"] is InvestigationReport

    system_prompt = agent_kwargs["system_prompt"]

    assert "highest_groups" in system_prompt
    assert "tied" in system_prompt.lower()
    assert "all groups" in system_prompt.lower()


    assert {
        tool.name for tool in agent_kwargs["tools"]
    } == {
        "profile_dataset_tool",
        "get_column_statistics_tool",
        "compare_groups_tool",
        "analyze_missing_values_tool",
        "analyze_correlation_tool",
    }

    assert len(agent_kwargs["middleware"]) == 1


def test_handle_tool_error_returns_guidance_for_value_error():
    request = SimpleNamespace(
        tool_call={
            "name": "compare_groups_tool",
        }
    )

    result = investigator.handle_tool_error(
        ValueError("Column not found"),
        request,
    )

    assert "compare_groups_tool" in result
    assert "invalid input" in result
    assert "correct the arguments" in result


def test_handle_tool_error_returns_none_for_other_exceptions():
    request = SimpleNamespace(
        tool_call={
            "name": "compare_groups_tool",
        }
    )

    result = investigator.handle_tool_error(
        RuntimeError("Unexpected failure"),
        request,
    )

    assert result is None
