from autonomous_data_detective.graph.graph import (
    build_investigation_graph,
)


def main():
    # Build the LangGraph workflow
    graph = build_investigation_graph()

    # Ask an investigation-only question
    initial_state = {
        "user_question": (
            "Which product has the highest total revenue?"
        ),
        "csv_path": "data/samples/sales.csv",
        "errors": [],
        "steps_taken": 0,
    }

    # Execute the workflow
    result = graph.invoke(initial_state)

    # Display results
    print("Workflow executed successfully.")
    print("\nSelected route:", result.get("route"))
    print("Steps taken:", result.get("steps_taken"))

    print("\nFinal response:\n")
    print(
        result.get(
            "final_response",
            "No final response generated.",
        )
    )

    # Verify the investigation-only route
    assert result.get("route") == "investigate", (
        "Expected the router to select 'investigate'."
    )

    assert not result.get("profile_result"), (
        "Profiler should not run for an investigation-only request."
    )

    assert result.get("investigation_result"), (
        "Investigation result is missing."
    )

    assert result.get("final_response"), (
        "Final response is missing."
    )

    assert result.get("steps_taken") == 1, (
        "Expected exactly one agent node to execute."
    )

    print("\nAll investigation-only assertions passed!")


if __name__ == "__main__":
    main()
