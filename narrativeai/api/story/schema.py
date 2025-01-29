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
    author: str | None = None

class StoryCreateRequestModel(BaseModel):
    title: str
    description: str
    genre_ids: list[str]
    cover_image: str | None = None
    author: str | None = None

class ListStoryResponseModel(BaseModel):
    stories: list[StoryModel]

class StoryStateModel(BaseModel):
    """Model for story state in database."""
    story_id: str
    stories: List[Message]
    longterm_plots: List[str]
    guidelines: List[str]
    requested_act: Optional[str]
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

class StoryCreateResponseModel(BaseModel):
    """Response model for story creation."""
    story_id: str



