from typing import Annotated, Tuple, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

MessageRole = Literal["user", "assistant", "system"]
Message = Tuple[MessageRole, str]

class GraphState(TypedDict):
    """State definition for the story generation workflow."""
    messages: Annotated[List[Message], add_messages]
