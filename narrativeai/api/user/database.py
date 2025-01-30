from datetime import datetime
import logging
from ..database import db_client
from ..dependencies import HttpExceptionCustom

logger = logging.getLogger(__name__)

def create_user(user_data: dict) -> str:
    """Create a new user in database."""
    logger.info(f"Creating user with firebase_uid: {user_data['firebase_uid']}")
    
    try:
        # Check if user already exists
        existing_user = db_client.user_collection.find_one({"firebase_uid": user_data["firebase_uid"]})
        if existing_user:
            logger.warning(f"User with firebase_uid {user_data['firebase_uid']} already exists")
            return user_data["firebase_uid"]
            
        # Add timestamps
        user_data["created_at"] = datetime.utcnow()
        user_data["updated_at"] = datetime.utcnow()
        
        # Insert user
        db_client.user_collection.insert_one(user_data)
        logger.info(f"Created user with firebase_uid: {user_data['firebase_uid']}")
        return user_data["firebase_uid"]
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HttpExceptionCustom.internal_server_error

def get_user_by_firebase_uid(firebase_uid: str) -> dict:
    """Get user from database by firebase UID."""
    logger.info(f"Fetching user by firebase_uid {firebase_uid}")
    
    try:
        user = db_client.user_collection.find_one({"firebase_uid": firebase_uid})
        if not user:
            logger.warning(f"No user found with firebase_uid {firebase_uid}")
            return None
            
        # Remove internal MongoDB ID
        if "_id" in user:
            del user["_id"]
            
        return user
        
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HttpExceptionCustom.internal_server_error

def update_user(firebase_uid: str, update_data: dict) -> bool:
    """Update user in database."""
    logger.info(f"Updating user {firebase_uid}")
    
    try:
        # Add updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        result = db_client.user_collection.update_one(
            {"firebase_uid": firebase_uid},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            logger.warning(f"No user updated with firebase_uid {firebase_uid}")
            return False
            
        logger.info(f"Updated user {firebase_uid}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HttpExceptionCustom.internal_server_error

def get_user_by_email(email: str) -> dict:
    """Get user from database by email."""
    logger.info(f"Fetching user by email {email}")
    
    try:
        user = db_client.user_collection.find_one({"email": email})
        if not user:
            logger.warning(f"No user found with email {email}")
            return None
            
        # Remove internal MongoDB ID
        if "_id" in user:
            del user["_id"]
            
        return user
        
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HttpExceptionCustom.internal_server_error 