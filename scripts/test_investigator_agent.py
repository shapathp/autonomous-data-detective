from autonomous_data_detective.agents.investigator import (
    build_investigator_agent,
)


def main():
    agent = build_investigator_agent()

    question = (
    "Using data/samples/sales.csv, calculate the average "
    "customer satisfaction score and explain whether it "
    "is related to revenue."
)

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                print(f"Tool called: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")

    print("\nFinal response:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
