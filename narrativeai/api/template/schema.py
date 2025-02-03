from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
from ..genre.schema import GenreModel

class TemplateListItemModel(BaseModel):
    """Model for template list items with limited fields."""
    id: str
    title: str
    description: str
    genre_list: List[GenreModel]
    cover_image: Optional[str] = None
    author: Optional[str] = None

class TemplateModel(BaseModel):
    """Model for template details."""
    id: str
    title: str
    description: str
    genre_list: List[GenreModel]
    initial_story: str
    params: Dict[str, str]
    cover_image: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class TemplateCreateRequestModel(BaseModel):
    """Model for template creation request."""
    title: str
    description: str
    genre_ids: List[str]
    initial_story: str
    params: Optional[Dict[str, str]] = None
    cover_image: Optional[str] = None
    author_firebase_uid: Optional[str] = None

class TemplateCreateResponseModel(BaseModel):
    """Model for template creation response."""
    template_id: str

class ListTemplateResponseModel(BaseModel):
    """Model for template list response."""
    templates: List[TemplateListItemModel] 