import json
from pathlib import Path

from autonomous_data_detective.agents.investigator import (
    build_investigator_agent,
)
from autonomous_data_detective.agents.schemas import InvestigationReport

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "samples" / "sales.csv"


def test_investigator_returns_structured_report_from_openai():
    """Inspect the real model response and tool execution."""

    agent = build_investigator_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Using the CSV dataset at "
                        f"{CSV_PATH}, identify which region has "
                        "the highest average revenue. Use the "
                        "appropriate analysis tool, report the "
                        "result, and explain any relevant limitation."
                    ),
                }
            ]
        }
    )

    report = result.get("structured_response")

    # Verify the structured response.
    assert isinstance(report, InvestigationReport)
    assert report.summary.strip()
    assert report.findings
    assert report.evidence
    assert isinstance(report.limitations, list)
    assert isinstance(report.recommendations, list)

    # Verify the key answer is correct.
    report_text = " ".join(
        [report.summary, *report.findings]
    ).lower()

    normalized_report = report_text.replace(",", "")

    assert "north" in normalized_report
    assert "56000" in normalized_report

    # Verify that the agent called the expected analysis tool.
    tool_calls = [
        tool_call
        for message in result["messages"]
        if message.type == "ai"
        for tool_call in message.tool_calls
    ]

    comparison_calls = [
        call
        for call in tool_calls
        if call["name"] == "compare_groups_tool"
    ]

    assert comparison_calls, (
        "The agent did not call compare_groups_tool."
    )

    comparison_call = comparison_calls[0]
    tool_args = comparison_call["args"]

    assert tool_args["group_by"].casefold() == "region"
    assert tool_args["metric"].casefold() == "revenue"
    assert tool_args["aggregation"] == "mean"

    # Verify the actual result returned by the tool.
    tool_messages = [
        message
        for message in result["messages"]
        if message.type == "tool"
        and message.name == "compare_groups_tool"
    ]

    assert tool_messages, (
        "No result was returned by compare_groups_tool."
    )

    tool_result = json.loads(tool_messages[0].content)

    assert tool_result["highest_group"] == "North"
    assert tool_result["highest_value"] == 56000.0
    assert tool_result["excluded_rows"] == 1
    assert tool_result["group_count"] == 4

    # Print the report for manual inspection.
    print("\n--- STRUCTURED INVESTIGATION REPORT ---")
    print(report.model_dump_json(indent=2))

    # Inspect the message history for tool calls and results.
    print("\n--- AGENT MESSAGE HISTORY ---")

    for message in result["messages"]:
        print(f"\nMessage type: {message.type}")

        if message.type == "ai":
            print("Content:", message.content)

            if message.tool_calls:
                print("Tool calls:", message.tool_calls)

        elif message.type == "tool":
            print("Tool name:", message.name)
            print("Tool result:", message.content)
