import logging
from datetime import datetime
from bson.objectid import ObjectId
from bson.errors import InvalidId

from ..database import db_client
from ..dependencies import HttpExceptionCustom
from .schema import StoryCreateRequestModel, StoryStateModel

logger = logging.getLogger(__name__)

def query_list_stories(skip: int, limit: int) -> list:
    try:
        stories_cursor = db_client.story_collection.find(skip=skip, limit=limit)
        stories = []
        for story in stories_cursor:
            story["id"] = str(story["_id"])
            genre_list = []
            for genre in story["genre_list"]:
                genre_list.append(str(genre))
            story["genre_list"] = genre_list
            stories.append(story)
        return stories

    except InvalidId:
        raise HttpExceptionCustom.bad_request

def query_list_genre() -> list:
    try:
        genres_cursor = db_client.genre_collection.find({})
        genres = []
        for genre in genres_cursor:
            genre["id"] = str(genre["_id"])
            genres.append(genre)
        return genres

    except InvalidId:
        raise HttpExceptionCustom.bad_request

async def query_story_state(story_id: str) -> dict:
    """Get story state from database."""
    logger.info(f"Fetching story state for story {story_id}")
    
    try:
        state_dict = db_client.story_states_collection.find_one({"story_id": ObjectId(story_id)})
        if not state_dict:
            logger.warning(f"No story state found for story {story_id}")
            return None

        # Convert story_id from ObjectId to string
        state_dict["story_id"] = str(state_dict["story_id"])
        
        # Convert stories array of arrays to array of tuples
        if "stories" in state_dict and state_dict["stories"]:
            state_dict["stories"] = [tuple(message) for message in state_dict["stories"]]
            
        return state_dict
        
    except Exception as e:
        logger.error(f"Error fetching story state: {e}")
        raise HttpExceptionCustom.internal_server_error

async def create_story_state(story_id: str, state: StoryStateModel) -> str:
    """Create story state in database."""
    logger.info(f"Creating story state for story {story_id}")
    state_dict = state.model_dump()
    state_dict["story_id"] = ObjectId(story_id)
    state_dict["updated_at"] = datetime.utcnow()
    
    try:
        insert_result = db_client.story_states_collection.insert_one(state_dict)
        return insert_result.inserted_id
    except Exception as e:
        logger.error(f"Error creating story state: {e}")
        raise HttpExceptionCustom.internal_server_error

async def update_story_state(story_id: str, state: StoryStateModel) -> bool:
    """Update story state in database."""
    logger.info(f"Updating story state for story {story_id}")
    
    try:
        # Update timestamp
        state.updated_at = datetime.utcnow()
        
        # Convert to dict and update
        state_dict = state.model_dump()
        state_dict["story_id"] = ObjectId(story_id)
        result = db_client.story_states_collection.update_one(
            {"story_id": ObjectId(story_id)},
            {"$set": state_dict},
            upsert=True
        )
        
        success = result.modified_count > 0 or result.upserted_id is not None
        if success:
            logger.info(f"Successfully updated story state for story {story_id}")
        else:
            logger.warning(f"No changes made to story state for story {story_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error updating story state: {e}")
        raise HttpExceptionCustom.internal_server_error

async def query_story(story_id: str) -> dict:
    """Get single story from database."""
    logger.info(f"Fetching story {story_id}")
    
    try:
        story = db_client.story_collection.find_one({"_id": ObjectId(story_id)})
        if not story:
            logger.warning(f"No story found with id {story_id}")
            return None
            
        # Convert _id to string id
        story["id"] = str(story["_id"])
        del story["_id"]
        
        # Convert genre ObjectIds to strings
        story["genre_list"] = [str(genre_id) for genre_id in story["genre_list"]]
        
        return story
        
    except Exception as e:
        logger.error(f"Error fetching story: {e}")
        raise HttpExceptionCustom.internal_server_error

async def create_story_doc(request: StoryCreateRequestModel):
    """Create a new story."""

    # Create story document
    story_doc = {
        "title": request.title,
        "description": request.description,
        "genre_list": [ObjectId(g_id) for g_id in request.genre_ids],
        "cover_image": request.cover_image,
        "author": request.author
    }

    # return story_id if successful
    insert_result = db_client.story_collection.insert_one(story_doc)
    return insert_result.inserted_id
