from datetime import datetime
import logging
from .database import create_user, get_user_by_firebase_uid, get_user_by_email, update_user
from .schema import UserModel, UserCreateRequestModel, UserUpdateRequestModel
from ..dependencies import HttpExceptionCustom

logger = logging.getLogger(__name__)

def create_user_response(request: UserCreateRequestModel) -> str:
    """Create a new user."""
    logger.info(f"Creating user with firebase_uid: {request.firebase_uid}")
    
    # Convert request model to dict
    user_data = request.model_dump()
    
    # Create user in database
    firebase_uid = create_user(user_data)
    if not firebase_uid:
        logger.error(f"Failed to create user with firebase_uid {request.firebase_uid}")
        raise HttpExceptionCustom.internal_server_error
        
    return firebase_uid

def get_user_by_firebase_response(firebase_uid: str) -> UserModel:
    """Get user by firebase UID."""
    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        raise HttpExceptionCustom.not_found("User not found")
    return UserModel(**user)

def update_user_response(firebase_uid: str, request: UserUpdateRequestModel) -> bool:
    """Update user."""
    # Convert request model to dict and remove None values
    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    
    if not update_data:
        logger.warning("No fields to update")
        return True
        
    success = update_user(firebase_uid, update_data)
    if not success:
        raise HttpExceptionCustom.not_found("User not found")
        
    return True

def get_user_by_email_response(email: str) -> UserModel:
    """Get user by email."""
    user = get_user_by_email(email)
    if not user:
        raise HttpExceptionCustom.not_found("User not found")
    return UserModel(**user) 