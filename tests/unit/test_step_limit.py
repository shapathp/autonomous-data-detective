import pytest

from autonomous_data_detective.graph.nodes import check_step_limit


class FakeSettings:
    max_investigation_steps = 8


@pytest.fixture
def fake_settings(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.get_settings",
        lambda: FakeSettings(),
    )


def test_step_limit_allows_execution_below_limit(fake_settings):
    state = {"steps_taken": 7}

    # No exception should be raised.
    check_step_limit(state)


@pytest.mark.parametrize("steps_taken", [8, 9, 20])
def test_step_limit_blocks_execution_at_or_above_limit(
    fake_settings, steps_taken
):
    state = {"steps_taken": steps_taken}

    with pytest.raises(
        RuntimeError,
        match="Maximum investigation steps reached: 8",
    ):
        check_step_limit(state)


def test_step_limit_defaults_to_zero_when_missing(fake_settings):
    state = {}

    # Missing steps_taken should be treated as zero.
    check_step_limit(state)
