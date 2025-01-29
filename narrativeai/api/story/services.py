from datetime import datetime
from .database import (
    create_story_doc,
    create_story_state,
    query_list_stories,
    query_list_genre,
    query_story_state,
    update_story_state,
    query_story
)

from .schema import StoryCreateRequestModel, StoryModel, GenreModel, StoryStateModel
from ...llm.states import Message
from ...llm.workflow import WorkflowBuilder
from ..dependencies import HttpExceptionCustom
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def get_message_content(message) -> Tuple[str, str]:
    """Extract content and role from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        return role, message.content
    elif isinstance(message, tuple):
        return message
    return "unknown", str(message)

def list_stories_response(skip: int, limit: int) -> list[StoryModel]:
    
    stories = query_list_stories(skip, limit)
    genre_lists = query_list_genre()
    
    # Create a lookup dictionary for faster genre name retrieval
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_lists}
    
    for story in stories:
        genre_list = []
        for genre_id in story["genre_list"]:
            genre_name = genre_lookup.get(genre_id, "Unknown")
            genre_list.append(GenreModel(id=genre_id, name=genre_name))
        story["genre_list"] = genre_list

    logger.info(f"Processed {len(stories)} stories with genres")
    return stories

async def get_story_response(story_id: str) -> StoryModel:
    story = await query_story(story_id)
    genre_lists = query_list_genre()
    
    # Create a lookup dictionary for faster genre name retrieval
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_lists}

    genre_list = []
    for genre_id in story["genre_list"]:
        genre_name = genre_lookup.get(genre_id, "Unknown")
        genre_list.append(GenreModel(id=genre_id, name=genre_name))
    story["genre_list"] = genre_list

    return story


async def get_story_state(story_id: str) -> StoryStateModel:
    """Get story state from database."""
    state_dict = await query_story_state(story_id)
    if state_dict:
        return StoryStateModel(**state_dict)
    return None

async def get_story_message(story_id: str) -> list[str]:
    """Get only the stories field from story state."""
    state = await get_story_state(story_id)
    if state and state.stories:
        return state.stories
    return []

async def write_story_message(story_id: str, message: str) -> List[Message]:
    """Process a user message through the LLM workflow and return new messages."""
    logger.info(f"Processing message for story {story_id}")
    
    # Get current state and story
    state = await get_story_state(story_id)
    story = await query_story(story_id)
    genre_list = query_list_genre()
    if not state or not story:
        logger.error(f"No state or story found for story {story_id}")
        return None
        
    # Add user message to state
    user_message: Message = ("user", message)
    state.stories.append(user_message)
    
    # Create genre name list
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_list}
    genre_names = [genre_lookup.get(genre_id, "Unknown") for genre_id in story["genre_list"]]
    
    # Get workflow with story's genre list
    workflow = WorkflowBuilder(genre_list=genre_names).compile()
    config = {"configurable": {"thread_id": story_id}}
    
    # Keep track of new messages
    new_messages = []
    last_story_len = len(state.stories)
    
    # Stream the updates
    last_event = None
    events = workflow.stream(state.model_dump(), config, stream_mode='values')
    for event in events:
        last_event = event

    story_messages = last_event.get("stories", [])
    if story_messages and len(story_messages) > last_story_len:
        role, content = get_message_content(story_messages[-1])
        if role != "user":
            new_messages.append((role, content))
        last_story_len = len(story_messages)
    
    # Convert messages to tuples for state model
    converted_messages = []
    for msg in story_messages:
        role, content = get_message_content(msg)
        converted_messages.append((role, content))

    guidelines = last_event.get("guidelines", [])
    guideline_contents = [g.content for g in guidelines]

    longterm_plots = last_event.get("longterm_plots", [])
    longterm_plot_contents = [p.content for p in longterm_plots]
    
    # Save final state
    state_model = StoryStateModel(
        story_id=story_id,
        stories=converted_messages,
        longterm_plots=longterm_plot_contents,
        guidelines=guideline_contents,
        requested_act=last_event.get("requested_act"),
        updated_at=datetime.utcnow()
    )
    await update_story_state(story_id, state_model)
    
    # Return only the new messages
    return new_messages

async def create_new_story(request: StoryCreateRequestModel) -> str:
    """Create a new story. Also initializes the story state."""
    
    # Create story document
    story_id = await create_story_doc(request)
    if not story_id:
        logger.error(f"Failed to create story document for story {request.title}")
        raise HttpExceptionCustom.internal_server_error

    # Initialize story state
    state_model = StoryStateModel(
        story_id=str(story_id),
        stories=[],
        longterm_plots=[],
        guidelines=[],
        requested_act=None,
        updated_at=datetime.utcnow()
    )

    # Save story state
    story_state_id = await create_story_state(story_id, state_model)
    if not story_state_id:
        logger.error(f"Failed to initialize story state for story {story_id}")
        raise HttpExceptionCustom.internal_server_error

    return str(story_id)
