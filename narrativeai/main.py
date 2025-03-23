#!/usr/bin/env python3

import os
import asyncio
import sys
import logging
from pprint import pprint
import uuid
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add proper import handling to support running directly or as a module
try:
    # When run as a module (python -m narrativeai.main)
    from .llm.workflow import WorkflowBuilder
    from .llm.states import GraphState
except ImportError as e:
    # When run directly (python main.py)
    logger.info(f"Adjusting import paths for direct execution. Error was: {str(e)}")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from narrativeai.llm.workflow import WorkflowBuilder
        from narrativeai.llm.states import GraphState
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        logger.error("Make sure you're in the correct directory and all dependencies are installed.")
        sys.exit(1)

from langgraph.graph.state import CompiledStateGraph
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# ANSI escape codes for colors
GRAY = "\033[90m"
BLUE = "\033[94m"
GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
RED = "\033[91m"
RESET = "\033[0m"

def get_message_content(message):
    """Extract content and role from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        return role, message.content
    elif isinstance(message, tuple):
        return message
    return "unknown", str(message)

async def process_user_input(user_input: str, workflow: CompiledStateGraph, genre_list: list, conversation_history=None, thread_id=None):
    """Process user input through the workflow.
    
    Args:
        user_input: User's input text
        workflow: Compiled workflow graph
        genre_list: List of genres for the story
        conversation_history: Previous conversation history to maintain context
        thread_id: Unique identifier for the conversation thread
        
    Returns:
        Final state after processing
    """
    try:
        # Initialize state
        initial_state = {
            "conversation_history": conversation_history or [],
            "current_message": HumanMessage(content=user_input),
            "memory_query": None,
            "memory_results": [],
            "genre_list": genre_list,
            "status": "RUNNING",
            "error": None
        }
        
        # Create config with thread_id for checkpointing
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        
        # Process the input with the config
        result = await workflow.ainvoke(initial_state, config)
        
        # Return the final state
        return result
    except Exception as e:
        logger.exception("Error in workflow processing")
        # Create an error state to return
        error_state = initial_state.copy() if 'initial_state' in locals() else {
            "conversation_history": conversation_history or [],
            "genre_list": genre_list,
        }
        error_state["error"] = f"Workflow error: {str(e)}"
        error_state["status"] = "ERROR"
        return error_state

def print_state_update(state: GraphState):
    """Print the state update in a formatted way."""
    # Print any errors
    if state.get("error"):
        error_msg = state["error"]
        print(f"{RED}Error: {error_msg}{RESET}")
        print(f"{GRAY}The system encountered an error but will continue.{RESET}")
        
    # Print the memory query if available
    if state.get("memory_query"):
        try:
            query_text = state["memory_query"].get("query_text", "Unknown query")
            print(f"{GRAY}Memory Query: {query_text}{RESET}")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Could not display memory query: {str(e)}")
            print(f"{GRAY}Memory Query: [malformed query]{RESET}")
        
    # Print memory results if available
    if state.get("memory_results"):
        mem_count = len(state["memory_results"])
        print(f"{GRAY}Retrieved {mem_count} memories{RESET}")
        
    # Print the last message in the conversation history
    history = state.get("conversation_history", [])
    if history:
        last_message = history[-1]
        if isinstance(last_message, AIMessage):
            print(f"{GREEN}Story:{RESET}")
            print(f"{last_message.content}")
        # If there's no AI message but there's an error, generate a helpful message
        elif state.get("error") and not any(isinstance(msg, AIMessage) for msg in history[-3:]):
            print(f"{GREEN}Story:{RESET}")
            print("The story pauses momentarily as the system processes your input...")

async def main_async():
    """Async main function."""
    genre_list = ["fantasy", "adventure", "mystery"]  # Example genre list
    
    print(f"{BLUE}Initializing NarrativeAI Interactive Story Assistant...{RESET}")
    
    try:
        # Create and compile the workflow
        workflow_builder = WorkflowBuilder(
            genre_list=genre_list,
            writer_model="gpt-4o",
            memory_model="gpt-3.5-turbo"
        )
        workflow = workflow_builder.compile()
        
        print(f"{BLUE}NarrativeAI Interactive Story Assistant{RESET}")
        print(f"{GRAY}Genre: {', '.join(genre_list)}{RESET}")
        print(f"{GRAY}Type 'exit' to quit{RESET}")
        print()
        
        # Keep track of conversation history between interactions
        conversation_history = []
        
        # Generate a unique thread ID for this session
        thread_id = str(uuid.uuid4())
        print(f"{GRAY}Session ID: {thread_id}{RESET}")
        
        # Interactive loop
        while True:
            try:
                user_input = input(f"{BLUE}User:{RESET} ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                    
                print(f"{GRAY}Processing...{RESET}")
                final_state = await process_user_input(
                    user_input, 
                    workflow, 
                    genre_list, 
                    conversation_history,
                    thread_id
                )
                print_state_update(final_state)
                
                # Update conversation history for next iteration
                # Only update if we have valid history in the state
                if final_state.get("conversation_history"):
                    conversation_history = final_state.get("conversation_history", [])
                    
                # Check if we need to reset due to critical error
                if final_state.get("status") == "CRITICAL_ERROR":
                    print(f"{RED}Critical error encountered. Resetting the system...{RESET}")
                    workflow = workflow_builder.compile()  # Recompile the workflow
                    conversation_history = []  # Reset history
                    thread_id = str(uuid.uuid4())  # New session
                    print(f"{GRAY}New Session ID: {thread_id}{RESET}")
                
                print()
                
            except KeyboardInterrupt:
                print("\nExiting gracefully...")
                break
            except Exception as e:
                logger.exception("Error processing input")
                error_details = traceback.format_exc()
                logger.error(f"Error details: {error_details}")
                print(f"{RED}Error: {str(e)}{RESET}")
                print(f"{ORANGE}The system encountered an error but will continue.{RESET}")
                print(f"{ORANGE}Please try again with different input or restart if problems persist.{RESET}")
    
    except Exception as e:
        logger.exception("Initialization error")
        print(f"{RED}Initialization error: {str(e)}{RESET}")
        print(f"{ORANGE}Make sure all dependencies are installed and configured properly.{RESET}")
        print(f"{ORANGE}Check the logs for more details.{RESET}")

def main():
    """Main entry point function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 