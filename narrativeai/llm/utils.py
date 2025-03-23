from typing import List
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from datetime import datetime, timezone

def format_conversation(messages: List[HumanMessage | AIMessage | SystemMessage]) -> str:
    """Format a list of conversation messages into a readable string.
    
    Args:
        messages: A list of LangChain message objects
        
    Returns:
        Formatted string representation of the conversation
    """
    if not messages:
        return "No conversation history available."
        
    formatted_parts = []
    
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_parts.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_parts.append(f"AI: {message.content}")
        elif isinstance(message, SystemMessage):
            formatted_parts.append(f"System: {message.content}")
        else:
            # Fallback for any other message types
            formatted_parts.append(f"Message: {str(message)}")
    
    return "\n\n".join(formatted_parts) 

def format_rfc3339_datetime(dt: datetime = None) -> str:
    """Format a datetime object to RFC3339 format with 'Z' timezone designator.
    
    This format is required by Weaviate for date fields.
    
    Args:
        dt: The datetime object to format. If None, uses current UTC time.
        
    Returns:
        RFC3339 formatted datetime string with 'Z' timezone designator
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        # Make naive datetime timezone-aware by assigning UTC
        dt = dt.replace(tzinfo=timezone.utc)
        
    # Format to ISO 8601 and replace +00:00 with Z for RFC3339 compliance
    return dt.isoformat().replace('+00:00', 'Z') 