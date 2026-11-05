import json

from autonomous_data_detective.tools.profiling import profile_dataset


def main() -> None:
    result = profile_dataset("data/samples/sales.csv")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
