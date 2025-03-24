import logging
from typing import Dict, List

import tiktoken

from narrativeai.llm.llm import ModelName, get_model, get_model_max_tokens

logger = logging.getLogger(__name__)

class MemoryAgent:
    """An agent that retrieve the story's memory based on few previous messages."""
    
    def __init__(self, genre_list: List[str], model_name: ModelName = "gpt-4o"):
        self.llm = get_model(model_name=model_name)
        self.genre_list = genre_list
        self.model_name = model_name
        self.max_context_tokens = get_model_max_tokens(model_name)
        
        # Initialize tiktoken encoder for token counting
        self.encoder = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's common encoder

    def _count_tokens(self, messages):
        """Count tokens in a list of messages using tiktoken."""
        num_tokens = 0
        for message in messages:
            # Count tokens in message content
            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, str):
                    num_tokens += len(self.encoder.encode(content))
            
            # Add a small fixed number for message structure overhead
            num_tokens += 4  # Approximate overhead per message
            
        return num_tokens

    async def ainvoke(self, state: Dict) -> str:
        """Invoke the memory agent to generate a memory query.
        
        Args:
            state: Current workflow state with conversation history
            
        Returns:
            String containing the structured memory query
        """
        # try:
            #messages = self._prepare_messages(state)
            
            # Call the LLM using ainvoke for async operations
            #response = await self.llm.ainvoke(messages)
            
            # Log and validate the response
            #content = response.content
            
            # Check if we got a valid response (basic validation)
            #if not content or not content.strip():
            #    raise ValueError("Empty response from memory agent")
                
            # if "query_text:" not in content:
            #     logger.warning(f"Response seems malformatted: {content}")
            #     # Attempt to fix basic issues - add query_text if missing
            #     content = f"query_text: context from recent messages\n{content}"
            
            # Return the memory query string
            # return content
            
        # except Exception as e:
        #     # Enhanced error handling with traceback
        #     logger.exception("Error in memory agent query generation")
        #     error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        #     logger.error(f"Detailed error: {error_details}")
            
        #     # Provide a fallback query that will at least return something
        #     fallback_query = "query_text: recent story events\nimportance_score_min: 2\nlimit: 3"
        #     logger.info(f"Using fallback query: {fallback_query}")
            
        #     # Return the fallback query instead of raising an exception
        #     # This allows the workflow to continue instead of failing
        #     return fallback_query
