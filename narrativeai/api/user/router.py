from fastapi import APIRouter
import logging
from .schema import UserModel, UserCreateRequestModel, UserUpdateRequestModel
from .services import (
    create_user_response,
    get_user_by_firebase_response,
    get_user_by_email_response,
    update_user_response
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/user", tags=["user"])

@router.post("", response_model=str)
def create_user(request: UserCreateRequestModel):
    """Create a new user."""
    return create_user_response(request)

@router.get("/{firebase_uid}", response_model=UserModel)
def get_user(firebase_uid: str):
    """Get user by firebase UID."""
    return get_user_by_firebase_response(firebase_uid)

@router.get("/email/{email}", response_model=UserModel)
def get_user_by_email(email: str):
    """Get user by email."""
    return get_user_by_email_response(email)

@router.patch("/{firebase_uid}", response_model=bool)
def update_user(firebase_uid: str, request: UserUpdateRequestModel):
    """Update user."""
    return update_user_response(firebase_uid, request) 