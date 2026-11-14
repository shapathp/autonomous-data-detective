from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import SecretStr

from autonomous_data_detective.agents import profiler
from autonomous_data_detective.tools.profiling import profile_dataset_tool


def test_build_profiler_agent_configures_model_and_tool(monkeypatch):
    settings = SimpleNamespace(
        openai_model="gpt-4.1-mini",
        openai_api_key=SecretStr("test-api-key"),
    )

    model = object()
    fake_agent = object()

    chat_openai_mock = MagicMock(return_value=model)
    create_agent_mock = MagicMock(return_value=fake_agent)

    monkeypatch.setattr(
        profiler,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        profiler,
        "ChatOpenAI",
        chat_openai_mock,
    )
    monkeypatch.setattr(
        profiler,
        "create_agent",
        create_agent_mock,
    )

    result = profiler.build_profiler_agent()

    assert result is fake_agent

    chat_openai_mock.assert_called_once_with(
        model="gpt-4.1-mini",
        api_key="test-api-key",
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    create_agent_mock.assert_called_once()

    agent_kwargs = create_agent_mock.call_args.kwargs

    assert agent_kwargs["model"] is model
    assert agent_kwargs["tools"] == [profile_dataset_tool]
    assert agent_kwargs["name"] == "profiler_agent"
    assert "Do not invent results." in agent_kwargs["system_prompt"]
    assert "correlation proves causation" in agent_kwargs["system_prompt"]
