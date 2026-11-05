from autonomous_data_detective.tools.profiling import (
    profile_dataset_tool,
)


def test_langchain_tool_has_name_and_schema():

    # assert statement checks whether a condition is true. the name is correct, the test passes. If it isn't, pytest reports a failure.
    assert profile_dataset_tool.name == "profile_dataset_tool"

    schema = profile_dataset_tool.args_schema.model_json_schema()

    # verifies that the schema exposes the input our tool needs.
    assert "csv_path" in schema["properties"]

# Check tool's actual execution using sample csv
def test_langchain_tool_runs_on_sample_csv():
    # run tool with given csv file
    result = profile_dataset_tool.invoke(
        {"csv_path": "data/samples/sales.csv"}
    )

    # assertion verify this stats matches or not
    assert result["row_count"] == 10
    assert result["column_count"] == 5
    assert result["duplicate_row_count"] == 1
    assert result["columns"]["revenue"]["missing_count"] == 1
