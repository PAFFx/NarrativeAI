from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from ...llm.states import Message

class GenreModel(BaseModel):
    id: str
    name: str

class StoryModel(BaseModel):
    id: str
    title: str
    description: str
    genre_list: list[GenreModel]
    cover_image: str | None = None
    author: str | None = None  # This will be the display name

class StoryCreateRequestModel(BaseModel):
    title: str
    description: str
    genre_ids: list[str]
    cover_image: str | None = None
    author_firebase_uid: str  # This is the Firebase UID for storage

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



