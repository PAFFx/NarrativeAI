from llm.workflow import WorkflowBuilder
from langgraph.graph.state import CompiledStateGraph
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from narrativeai.llm.models import StoryContext

def get_message_content(message):
    """Extract content and role from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        return role, message.content
    elif isinstance(message, tuple):
        return message
    return "unknown", str(message)

def stream_graph_updates(user_input: str, workflow: CompiledStateGraph):
    config = {"configurable": {"thread_id": "1"}}
    events = workflow.stream({"messages": [("user", user_input)]}, config, stream_mode='values')
    for event in events:
        messages = event["messages"]
        if messages:
            role, content = get_message_content(messages[-1])
            if (role != "user"):
                print(f"{role.capitalize()}: {content}")


def main():
    story_context = StoryContext(genre="mecha", tone="war")
    workflow = WorkflowBuilder(story_context).compile()
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, workflow)
        except Exception as e:
            print("Exception:", e)
            break

if __name__ == "__main__":
    main() 