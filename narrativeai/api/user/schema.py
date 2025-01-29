from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserModel(BaseModel):
    """Model for user in database."""
    firebase_uid: str
    email: str
    display_name: Optional[str] = None
    photo_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class UserCreateRequestModel(BaseModel):
    """Model for user creation request."""
    firebase_uid: str
    email: str
    display_name: Optional[str] = None
    photo_url: Optional[str] = None

class UserUpdateRequestModel(BaseModel):
    """Model for user update request."""
    display_name: Optional[str] = None
    photo_url: Optional[str] = None 