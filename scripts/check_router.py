from autonomous_data_detective.graph.router import route_question


def main():
    test_cases = [
        (
            "Give me a profile of the dataset.",
            "profile",
        ),
        (
            "Which product has the highest total revenue?",
            "investigate",
        ),
        (
            "Profile the dataset and calculate total revenue by product.",
            "both",
        ),
    ]

    for question, expected_route in test_cases:
        result = route_question(
            {"user_question": question}
        )

        actual_route = result["route"]

        print(f"Question: {question}")
        print(f"Expected: {expected_route}")
        print(f"Actual:   {actual_route}")

        assert actual_route == expected_route, (
            f"Expected {expected_route}, got {actual_route}"
        )

        print("PASS\n")

    print("All router tests passed.")


if __name__ == "__main__":
    main()
