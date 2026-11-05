from autonomous_data_detective.core.config import get_settings
from autonomous_data_detective.graph.nodes import check_step_limit


def main():
    settings = get_settings()
    max_steps = settings.max_investigation_steps

    print("Configured maximum steps:", max_steps)

    # Test 1: Below the limit should be allowed
    state_below_limit = {
        "steps_taken": max_steps - 1,
    }

    check_step_limit(state_below_limit)
    print("PASS: Execution is allowed below the limit.")

    # Test 2: At the limit should raise an error
    state_at_limit = {
        "steps_taken": max_steps,
    }

    try:
        check_step_limit(state_at_limit)
    except RuntimeError as exc:
        print("PASS: Step limit correctly blocks execution.")
        print("Error message:", exc)
    else:
        raise AssertionError(
            "Expected RuntimeError when the step limit is reached."
        )

    print("\nAll step-limit tests passed!")


if __name__ == "__main__":
    main()
