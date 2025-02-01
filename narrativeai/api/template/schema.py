from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
from ..genre.schema import GenreModel

class TemplateModel(BaseModel):
    """Model for template in database."""
    id: str
    title: str
    description: str
    genre_list: list[GenreModel]  # Use GenreModel to include genre names
    initial_story: str
    params: Dict[str, str] = {}  # Store parameters and their default values
    cover_image: str | None = None
    author: str | None = None
    created_at: datetime = None
    updated_at: datetime = None

class TemplateCreateRequestModel(BaseModel):
    """Model for template creation request."""
    title: str
    description: str
    genre_ids: list[str]
    initial_story: str
    params: Dict[str, str] = {}  # Parameters extracted from initial_story with default values
    cover_image: str | None = None
    author_firebase_uid: str

class ListTemplateResponseModel(BaseModel):
    """Response model for listing templates."""
    templates: list[TemplateModel]

class TemplateCreateResponseModel(BaseModel):
    """Response model for template creation."""
    template_id: str 