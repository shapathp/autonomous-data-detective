import pytest

from autonomous_data_detective.graph.router import route_question


@pytest.mark.parametrize(
    ("question", "expected_route"),
    [
        ("Give me a profile of the dataset.", "profile"),
        ("Show me the dataset structure.", "profile"),
        ("Find missing values in the dataset.", "profile"),
        ("Which product has the highest revenue?", "investigate"),
        ("Calculate the average revenue.", "investigate"),
        ("Compare revenue by region.", "investigate"),
        (
            "Profile the dataset and calculate total revenue by product.",
            "both",
        ),
        (
            "Show missing values and identify the highest-revenue product.",
            "both",
        ),
        (
            "What is the overall mean salary across all rows?",
            "investigate",
        ),
        (
            "Calculate the median revenue.",
            "investigate",
        ),
        (
            "Calculate the standard deviation of salary.",
            "investigate",
        ),
        (
            "Show missing values and calculate the median salary.",
            "both",
        ),
    ],
)
def test_route_question_selects_expected_route(
    question,
    expected_route,
):
    result = route_question({"user_question": question})

    assert result["route"] == expected_route


def test_unclear_question_routes_to_clarify():
    result = route_question(
        {"user_question": "Hello, how are you?"}
    )
    assert result["route"] == "clarify"


def test_greeting_is_not_treated_as_a_valid_investigation():
    result = route_question({"user_question": "Hello, how are you?"})
    assert result["route"] == "clarify"


def test_empty_question_requests_clarification():
    result = route_question({"user_question": ""})
    assert result["route"] == "clarify"


@pytest.mark.parametrize(
    "question",
    [
        "What is the minimum salary?",
        "What is the maximum salary?",
        "What is the smallest revenue?",
        "What is the largest revenue?",
        "What is the least salary?",
        "What is the greatest revenue?",
        "What is the min salary?",
        "What is the max salary?",
    ],
)
def test_route_numerical_extreme_questions_to_investigate(question):
    result = route_question({"user_question": question})

    assert result["route"] == "investigate"


@pytest.mark.parametrize(
    ("question", "expected_route"),
    [
        # Different ways to request dataset profiling
        ("Summarize the dataset structure.", "profile"),
        ("Give me an overview of the data quality.", "profile"),
        ("Which columns contain missing values?", "profile"),
        ("Are there duplicate records?", "profile"),

        # Different ways to request numerical analysis
        ("How much revenue was generated on average?", "investigate"),
        ("Find the middle salary value.", "investigate"),
        ("Identify the department with the greatest salary.", "investigate"),
        ("Measure the relationship between salary and revenue.", "investigate"),

        # Combined requests
        (
            "Summarize the dataset and calculate average revenue.",
            "both",
        ),
        (
            "Check data quality and compare average salary by department.",
            "both",
        ),
    ],
)
def test_router_handles_varied_question_phrasings(
    question, expected_route
):
    result = route_question({"user_question": question})

    assert result["route"] == expected_route
