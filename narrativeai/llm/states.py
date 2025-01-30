from typing import Annotated, Tuple, List, Literal, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dataclasses import dataclass
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

MessageRole = Literal["user", "assistant", "system"]
Message = Tuple[MessageRole, str]

class GraphState(TypedDict):
    """State definition for the story generation workflow."""
    stories: Annotated[List[Message], add_messages]
    longterm_plots: Annotated[List[str], add_messages]
    guidelines: Annotated[List[str], add_messages]
    requested_act: Optional[str]
    conseq_longterm_count: int  # Track consecutive longterm plotter invocations