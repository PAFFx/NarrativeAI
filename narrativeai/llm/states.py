from typing import Annotated, Tuple, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from .models import StoryContext

MessageRole = Literal["user", "assistant", "system"]
Message = Tuple[MessageRole, str]

def UpdateStoryContext(old: StoryContext, new: StoryContext):
    """Update the story context with new values only if new values are provided."""
    ctx = StoryContext(
        genre=new.genre if new.genre else old.genre,
        tone=new.tone if new.tone else old.tone
    )
    return ctx

class GraphState(TypedDict):
    """State definition for the story generation workflow."""
    stories: Annotated[List[Message], add_messages]
    longterm_plots: Annotated[List[str], add_messages]
    context: Annotated[StoryContext, UpdateStoryContext]