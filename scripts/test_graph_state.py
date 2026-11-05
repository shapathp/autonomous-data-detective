from autonomous_data_detective.graph.state import InvestigationState


def main():
    state: InvestigationState = {
        "user_question": "Which product has the highest total revenue?",
        "csv_path": "data/samples/sales.csv",
        "route": "investigate",
        "errors": [],
        "steps_taken": 0,
    }

    print("State created successfully.")
    print(state)


if __name__ == "__main__":
    main()
