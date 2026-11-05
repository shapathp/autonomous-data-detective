from autonomous_data_detective.graph.nodes import profile_node


def main():
    initial_state = {
        "user_question": "Profile the sales dataset.",
        "csv_path": "data/samples/sales.csv",
        "errors": [],
        "steps_taken": 0,
    }

    result = profile_node(initial_state)

    print("Profiler node executed successfully.")
    print("\nSteps taken:", result["steps_taken"])
    print("\nProfile result:\n")
    print(result["profile_result"])


if __name__ == "__main__":
    main()
