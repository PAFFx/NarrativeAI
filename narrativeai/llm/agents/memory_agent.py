from typing import Dict, List
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from ..utils import format_conversation
from ..llm import get_model, ModelName, get_model_max_tokens
from langchain_core.messages.utils import trim_messages
import logging
import traceback
import tiktoken

logger = logging.getLogger(__name__)

class MemoryAgent:
    """An agent that retrieve the story's memory based on few previous messages."""
    
    def __init__(self, genre_list: List[str], model_name: ModelName = "gpt-3.5-turbo"):
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

    def _prepare_messages(self, state: Dict) -> List[SystemMessage | HumanMessage]:
        """Prepare the messages for the memory agent."""
        messages = []
        
        # Extract conversation history from state
        history = state.get("conversation_history", [])
        
        # Get the last 3 messages if available
        last_messages = history[-3:] if len(history) >= 3 else history
        
        # Format the extracted messages
        formatted_messages = format_conversation(last_messages)
        
        # Create system message with instructions for the memory agent
        system_prompt = """You are a Memory Agent for an interactive storytelling system. 
Your role is to analyze recent conversation messages and extract key information to retrieve relevant story memories.

Based on the last few messages, generate a structured query to retrieve memories from a story database.
Your output will be used to query a Weaviate vector database with the following schema:

```
class StoryMemory {{
    text: text
    characters: text[]
    locations: text[]
    themes: text[]
    timestamp: date
    chapter: text
    importance_score: number
    storyline: text
    emotion: text
}}
```

IMPORTANT: You must format your response exactly as shown below:
query_text: <main search concept>
characters: <comma separated list or empty>
locations: <comma separated list or empty>
themes: <comma separated list or empty>
importance_score_min: <number between 0-10>
storyline: <specific storyline or empty>
limit: <number of results, default 5>

Genre of the story: {genres}

Remember: Focus on extracting narrative elements that are most likely to have relevant memories associated with them.
"""
        
        # Format the system prompt with the genre list
        formatted_system_prompt = system_prompt.format(genres=", ".join(self.genre_list))
        
        # Create the final messages list
        messages.append(SystemMessage(content=formatted_system_prompt))
        messages.append(HumanMessage(content=f"Recent conversation:\n{formatted_messages}\n\nBased on these messages, generate a query to retrieve relevant memories."))
        
        # Trim messages if necessary (although this is a simple conversation)
        max_tokens = self.max_context_tokens // 2
        trimmed_messages = trim_messages(
            messages,
            max_tokens=max_tokens,
            token_counter=self._count_tokens,
            strategy="last",
            include_system=True
        )
        
        if isinstance(trimmed_messages, dict) and 'truncated' in trimmed_messages:
            return trimmed_messages['truncated']
        return trimmed_messages

    async def ainvoke(self, state: Dict) -> str:
        """Invoke the memory agent to generate a memory query.
        
        Args:
            state: Current workflow state with conversation history
            
        Returns:
            String containing the structured memory query
        """
        try:
            messages = self._prepare_messages(state)
            
            # Call the LLM using ainvoke for async operations
            response = await self.llm.ainvoke(messages)
            
            # Log and validate the response
            content = response.content
            
            # Check if we got a valid response (basic validation)
            if not content or not content.strip():
                raise ValueError("Empty response from memory agent")
                
            if "query_text:" not in content:
                logger.warning(f"Response seems malformatted: {content}")
                # Attempt to fix basic issues - add query_text if missing
                content = f"query_text: context from recent messages\n{content}"
            
            # Return the memory query string
            return content
            
        except Exception as e:
            # Enhanced error handling with traceback
            logger.exception("Error in memory agent query generation")
            error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Detailed error: {error_details}")
            
            # Provide a fallback query that will at least return something
            fallback_query = "query_text: recent story events\nimportance_score_min: 2\nlimit: 3"
            logger.info(f"Using fallback query: {fallback_query}")
            
            # Return the fallback query instead of raising an exception
            # This allows the workflow to continue instead of failing
            return fallback_query
