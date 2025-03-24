from pydantic import BaseModel
from typing import List, Optional, Dict, Literal
from datetime import datetime
from ...llm.states import Message
from enum import Enum

from ..genre.schema import GenreModel

class StoryModel(BaseModel):
    id: str
    title: str
    description: str
    genre_list: list[GenreModel]
    cover_image: str | None = None
    author: str | None = None  # This will be the display name
    template_id: str | None = None

class StoryCreateRequestModel(BaseModel):
    title: str
    description: str
    genre_ids: list[str]
    cover_image: str | None = None
    author_firebase_uid: str  # This is the Firebase UID for storage
    template_id: str | None = None

class ListStoryResponseModel(BaseModel):
    stories: list[StoryModel]

class StoryStateModel(BaseModel):
    """Model for story state in database."""
    story_id: str
    stories: List[Message]
    longterm_plots: List[str]
    guidelines: List[str]
    requested_act: Optional[str]
    conseq_longterm_count: int = 0  # Track consecutive longterm plotter invocations
    updated_at: datetime = None
    thread_id: Optional[str] = None  # Added for memory isolation

    class Config:
        json_encoders = {
            Message: lambda m: {"role": m[0], "content": m[1]} if isinstance(m, tuple) else {"role": "assistant", "content": m.content}
        }

class StoryMessageModel(BaseModel):
    messages: List[Message]

class WriteRequestModel(BaseModel):
    """Model for write request body."""
    message: str
    model: str | None = "gpt-4o"  # Available models: gpt-4o, claude-3-sonnet

class StoryCreateResponseModel(BaseModel):
    """Response model for story creation."""
    story_id: str

class WriteFromPromptRequestModel(BaseModel):
    """Model for write from prompt request body."""
    story: str
    model: str | None = "gpt-4o"  # Available models: gpt-4o, claude-3-sonnet

class WriteFromPromptResponseModel(BaseModel):
    """Response model for write from prompt."""
    next_story: str

class StoryFromTemplateRequestModel(BaseModel):
    """Model for creating story from template."""
    template_id: str
    params: Dict[str, str]
    author_firebase_uid: Optional[str] = None

class StoryUpdateRequestModel(BaseModel):
    """Model for updating story details."""
    title: Optional[str] = None
    description: Optional[str] = None

class MessageOperation(str, Enum):
    """Enum for message operations."""
    EDIT = "edit"
    DELETE = "delete"
    INSERT = "insert"

class MessageEditItem(BaseModel):
    """Model for a single message edit operation."""
    index: int
    content: Optional[str] = None  # Optional for delete operation
    operation: MessageOperation
    role: Optional[str] = "user"  # Only used for insert operation

class MessageEditRequestModel(BaseModel):
    """Model for editing story messages."""
    messages: List[MessageEditItem]
