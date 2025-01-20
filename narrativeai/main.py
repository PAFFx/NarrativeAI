from pprint import pprint
from .llm.workflow import WorkflowBuilder
from langgraph.graph.state import CompiledStateGraph
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# ANSI escape codes for colors
GRAY = "\033[90m"
BLUE = "\033[94m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

def get_message_content(message):
    """Extract content and role from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        return role, message.content
    elif isinstance(message, tuple):
        return message
    return "unknown", str(message)

def stream_graph_updates(user_input: str, workflow: CompiledStateGraph, config: dict):
    # Initialize state
    initial_state = {
        "stories": [("user", user_input)],
        "longterm_plots": [],
        "guidelines": [],
        "requested_act": None
    }
    
    # Keep track of what we've seen
    last_plot_len = 0
    last_story_len = 1  # Start at 1 because we have the initial user input
    last_guidelines_len = 0
    
    events = workflow.stream(initial_state, config, stream_mode='values')
    for event in events:
        # Check and print new plots
        longterm_plots = event.get("longterm_plots", [])
        if longterm_plots and len(longterm_plots) > last_plot_len:
            _, content = get_message_content(longterm_plots[-1])
            print(f"{GRAY}{content}{RESET}")
            last_plot_len = len(longterm_plots)

        # Check and print new guidelines
        guidelines = event.get("guidelines", [])
        if guidelines and len(guidelines) > last_guidelines_len:
            print(f"{ORANGE}Guidelines: {guidelines[-1]}{RESET}")
            last_guidelines_len = len(guidelines)

        # Check and print new story entries
        story = event.get("stories", [])
        if story and len(story) > last_story_len:
            role, content = get_message_content(story[-1])
            if role != "user":
                print(f"{BLUE}{role.capitalize()}: {content}{RESET}")
            last_story_len = len(story)

def main():
    config = {"configurable": {"thread_id": "1"}}
    genre_list = ["mecha", "war", "sci-fi"]  # Example genre list
    workflow = WorkflowBuilder(genre_list=genre_list).compile()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, workflow, config)
        except Exception as e:
            print("Exception:", e)
            break

if __name__ == "__main__":
    main() 