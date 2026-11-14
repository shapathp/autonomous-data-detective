from autonomous_data_detective.agents.profiler import build_profiler_agent


def main() -> None:
    agent = build_profiler_agent()

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    (
                        "Profile data/samples/sales.csv. Report the dataset "
                        "dimensions, duplicate rows, missing values, and "
                        "important observations."
                    ),
                )
            ]
        }
    )

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"\nTool requested: {call['name']}")
                print(f"Tool arguments: {call['args']}")

    print("\n--- Final Profiler Agent Response ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
