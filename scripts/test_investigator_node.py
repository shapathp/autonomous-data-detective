from autonomous_data_detective.graph.nodes import (
    investigate_node,
)


def main():
    initial_state = {
        "user_question": (
            "Which product has the highest total revenue?"
        ),
        "csv_path": "data/samples/sales.csv",
        "errors": [],
        "steps_taken": 0,
    }

    result = investigate_node(initial_state)

    print("Investigator node executed successfully.")
    print("\nSteps taken:", result["steps_taken"])
    print("\nInvestigation result:\n")
    print(result["investigation_result"])


if __name__ == "__main__":
    main()
