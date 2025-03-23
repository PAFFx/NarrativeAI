from typing import Annotated, Tuple, List, Literal, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dataclasses import dataclass
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

class GraphState(TypedDict):
    """State definition for the story generation workflow."""
    conversation_history: List[HumanMessage | AIMessage | SystemMessage]
    current_message: Optional[Union[HumanMessage, AIMessage, SystemMessage]]
    memory_query: Optional[dict]
    memory_results: Optional[List[dict]]
    genre_list: List[str]
    status: Literal["RUNNING", "WAITING_USER_INPUT", "COMPLETE"]
    error: Optional[str]