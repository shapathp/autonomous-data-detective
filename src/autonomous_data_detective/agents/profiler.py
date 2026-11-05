from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from autonomous_data_detective.core.config import get_settings
from autonomous_data_detective.tools.profiling import profile_dataset_tool


def build_profiler_agent():
    """Build an agent that can inspect CSV datasets using the profiler tool."""
    settings = get_settings()

    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    return create_agent(
        model=model,
        tools=[profile_dataset_tool],
        system_prompt=(
            "You are the Profiler Agent in an autonomous data investigation "
            "system. Use profile_dataset_tool when the user asks you to "
            "inspect a dataset. Base all numerical claims on tool output. "
            "Summarize row count, column count, missing values, duplicates, "
            "and notable column statistics. Do not invent results. "
            "Do not claim that correlation proves causation."
        ),
        name="profiler_agent",
    )
