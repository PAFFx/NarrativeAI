from typing import Annotated, Tuple, List, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dataclasses import dataclass

MessageRole = Literal["user", "assistant", "system"]
Message = Tuple[MessageRole, str]

@dataclass
class GraphState(TypedDict):
    """State definition for the story generation workflow."""
    stories: Annotated[List[Message], add_messages]
    longterm_plots: Annotated[List[str], add_messages]
    guidelines: Annotated[List[str], add_messages]
    genre: str
    requested_act: Optional[str]