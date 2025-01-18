from pprint import pprint
from llm.workflow import WorkflowBuilder
from langgraph.graph.state import CompiledStateGraph
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from llm.models import StoryContext

# ANSI escape codes for colors
GRAY = "\033[90m"
BLUE = "\033[94m"
RESET = "\033[0m"

def get_message_content(message):
    """Extract content and role from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        return role, message.content
    elif isinstance(message, tuple):
        return message
    return "unknown", str(message)

def stream_graph_updates(user_input: str, workflow: CompiledStateGraph, story_context: StoryContext, config: dict):
    events = workflow.stream({"stories": [("user", user_input)], "context": story_context}, config, stream_mode='values')
    for event in events:
        longterm_plots = event["longterm_plots"]
        if longterm_plots and len(longterm_plots) != 0:
            _, content = get_message_content(longterm_plots[-1])
            print(f"{GRAY}{content}{RESET}")

        story = event["stories"]
        if story and len(story) != 0:
            role, content = get_message_content(story[-1])
            if (role != "user"):
                print(f"{BLUE}{role.capitalize()}: {content}{RESET}")


def main():
    config = {"configurable": {"thread_id": "1"}}
    story_context = StoryContext(genre="mecha", tone="war")
    workflow = WorkflowBuilder(story_context).compile()
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, workflow, story_context, config)
        except Exception as e:
            print("Exception:", e)
            break

if __name__ == "__main__":
    main() 