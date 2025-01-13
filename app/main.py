from llm.workflow import WorkflowBuilder
from langgraph.graph.state import CompiledStateGraph

def stream_graph_updates(user_input: str, workflow: CompiledStateGraph):
    for event in workflow.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def main():
    workflow = WorkflowBuilder().compile()
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, workflow)
        except Exception as e:
            print("Exception:",e)
            break

if __name__ == "__main__":
    main() 