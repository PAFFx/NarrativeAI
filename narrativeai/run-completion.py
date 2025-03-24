#!/usr/bin/env python3

import os
import asyncio
import sys
import logging
import uuid
import json
from typing import List, Dict, Any, Optional
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add proper import handling to support running directly or as a module
try:
    # When run as a module (python -m narrativeai.run-completion)
    from .llm.workflow import WorkflowBuilder
    from .llm.states import GraphState
    from .llm.llm import get_model_name
    from langchain.schema import HumanMessage
except ImportError as e:
    # When run directly (python run-completion.py)
    logger.info(f"Adjusting import paths for direct execution. Error was: {str(e)}")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from narrativeai.llm.workflow import WorkflowBuilder
        from narrativeai.llm.states import GraphState
        from narrativeai.llm.llm import get_model_name
        from langchain.schema import HumanMessage
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        logger.error("Make sure you're in the correct directory and all dependencies are installed.")
        sys.exit(1)


async def process_prompt(prompt: str, workflow, genre_list: List[str]) -> str:
    """Process a single prompt through the workflow.
    
    Args:
        prompt: The story prompt to process
        workflow: Compiled workflow graph
        genre_list: List of genres for the story
    
    Returns:
        Generated story response
    """
    try:
        # Generate a unique thread ID for this prompt
        thread_id = str(uuid.uuid4())
        logger.info(f"Processing prompt with thread ID: {thread_id}")
        
        # Initialize state with the prompt as user message
        initial_state = {
            "conversation_history": [],
            "current_message": HumanMessage(content=prompt),
            "memory_query": None,
            "memory_results": [],
            "genre_list": genre_list,
            "status": "RUNNING",
            "error": None,
            "thread_id": thread_id
        }
        
        # Process through workflow with thread-specific config
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke workflow to process the prompt
        result = await workflow.ainvoke(initial_state, config)
        
        # Get response from conversation history
        history = result.get("conversation_history", [])
        if not history:
            return "Error: No response generated."
        
        # Get the last AI response
        for msg in reversed(history):
            if hasattr(msg, "type") and msg.type == "ai":
                return msg.content
            if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                return msg.content
        
        return "Error: No AI response found in conversation history."
        
    except Exception as e:
        logger.exception(f"Error processing prompt: {str(e)}")
        return f"Error generating response: {str(e)}"

async def generate_completions(
    prompts: List[str], 
    writer_model: str = "gpt-4o", 
    memory_model: str = "gpt-3.5-turbo", 
    genres: List[str] = None
) -> List[str]:
    """Generate completions for multiple prompts.
    
    Args:
        prompts: List of prompts to process
        writer_model: Model name to use for writing (default: gpt-4o)
        memory_model: Model name to use for memory retrieval (default: gpt-3.5-turbo)
        genres: List of genres for the stories (default: creative, fantasy)
    
    Returns:
        List of generated story responses
    """
    # Set default genres if none provided
    if not genres:
        genres = ["creative"]
    
    logger.info(f"Generating completions for {len(prompts)} prompts")
    logger.info(f"Writer model: {writer_model}, Memory model: {memory_model}, Genres: {genres}")
    
    try:
        # Convert user-friendly model names to actual model names
        writer_model_name = get_model_name(writer_model)
        memory_model_name = get_model_name(memory_model)
        
        # Create and compile the workflow with separate models
        workflow_builder = WorkflowBuilder(
            genre_list=genres,
            writer_model=writer_model_name,
            memory_model=memory_model_name
        )
        workflow = workflow_builder.compile()
        
        # Process each prompt
        tasks = [process_prompt(prompt, workflow, genres) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        return responses
        
    except Exception as e:
        logger.exception(f"Error in generation process: {str(e)}")
        # Return error messages for all prompts
        return [f"Error: {str(e)}"] * len(prompts)


async def main_async():
    """Async main function for CLI usage."""
    parser = ArgumentParser(description="Generate story completions from prompts")
    parser.add_argument("--prompts", type=str, help="JSON string or file path containing list of prompts")
    parser.add_argument("--writer-model", type=str, default="gpt-4o", help="Writer model to use (default: gpt-4o)")
    parser.add_argument("--memory-model", type=str, default="gpt-3.5-turbo", help="Memory model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--genres", type=str, default="creative,fantasy", help="Comma-separated list of genres")
    parser.add_argument("--output", type=str, help="Output file path (default: print to stdout)")
    
    args = parser.parse_args()
    
    if not args.prompts:
        parser.error("Please provide prompts using --prompts")

    # Parse genres
    genres = [genre.strip() for genre in args.genres.split(",")]
    
    # Parse prompts (either JSON string or file path)
    try:
        if os.path.isfile(args.prompts):
            with open(args.prompts, 'r') as f:
                prompts = json.load(f)
        else:
            prompts = json.loads(args.prompts)
            
        if not isinstance(prompts, list):
            prompts = [prompts]
            
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error parsing prompts: {str(e)}")
        sys.exit(1)
    
    # Generate completions with separate models
    responses = await generate_completions(
        prompts=prompts, 
        writer_model=args.writer_model,
        memory_model=args.memory_model,
        genres=genres
    )
    
    # Output results
    result = {"responses": responses}
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))

def main():
    """Main entry point function."""
    asyncio.run(main_async())


# Python API for programmatic usage
def run_completions(
    prompts: List[str], 
    writer_model: str = "gpt-4o", 
    memory_model: str = "gpt-3.5-turbo", 
    genres: List[str] = None
) -> List[str]:
    """Generate story completions from a list of prompts.
    
    Args:
        prompts: List of prompts to generate stories from
        writer_model: Model to use for writing (default: gpt-4o)
        memory_model: Model to use for memory retrieval (default: gpt-3.5-turbo)
        genres: List of genres (default: creative, fantasy)
    
    Returns:
        List of generated story responses
    """
    return asyncio.run(generate_completions(prompts, writer_model, memory_model, genres))


if __name__ == "__main__":
    main()
