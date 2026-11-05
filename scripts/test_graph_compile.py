from autonomous_data_detective.graph.graph import (
    build_investigation_graph,
)


def main():
    graph = build_investigation_graph()

    print("LangGraph compiled successfully.")
    print("\nGraph structure:\n")
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
