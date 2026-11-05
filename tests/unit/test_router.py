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
