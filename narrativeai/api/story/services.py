from datetime import datetime

from narrativeai.llm.llm import ModelName
from .database import (
    create_story_doc,
    create_story_state,
    query_list_stories,
    query_list_genre,
    query_story_state,
    update_story_state,
    query_story,
    delete_story,
    update_story
)

from .schema import MessageEditItem, MessageOperation, StoryCreateRequestModel, StoryModel, GenreModel, StoryStateModel, StoryFromTemplateRequestModel, StoryUpdateRequestModel
from ...llm.states import Message
from ...llm.workflow import WorkflowBuilder
from ...llm.llm import get_model_name
from ..dependencies import HttpExceptionCustom
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import logging
from typing import List, Tuple, Dict
from ..user.database import get_user_by_firebase_uid
from langchain.prompts import ChatPromptTemplate
from ..template.services import get_template_response

logger = logging.getLogger(__name__)

def get_message_content(message) -> Tuple[str, str]:
    """Extract content and role from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        return role, message.content
    elif isinstance(message, tuple):
        return message
    return "unknown", str(message)

def list_stories_response(skip: int, limit: int, author_firebase_uid: str | None = None) -> list[StoryModel]:
    """List stories with optional author filter."""
    stories = query_list_stories(skip, limit, author_firebase_uid)
    genre_lists = query_list_genre()
    
    # Create a lookup dictionary for faster genre name retrieval
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_lists}
    
    # Process each story
    for story in stories:
        # Process genres
        genre_list = []
        for genre_id in story["genre_list"]:
            genre_name = genre_lookup.get(genre_id, "Unknown")
            genre_list.append(GenreModel(id=genre_id, name=genre_name))
        story["genre_list"] = genre_list
        
        # Get author display name
        if "author_firebase_uid" in story:
            user = get_user_by_firebase_uid(story["author_firebase_uid"])
            if user:
                story["author"] = user.get("display_name", None)
            else:
                story["author"] = None
            del story["author_firebase_uid"]

    logger.info(f"Processed {len(stories)} stories with genres and authors")
    return stories

def get_story_response(story_id: str) -> StoryModel:
    story = query_story(story_id)
    genre_lists = query_list_genre()
    
    # Create a lookup dictionary for faster genre name retrieval
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_lists}

    genre_list = []
    for genre_id in story["genre_list"]:
        genre_name = genre_lookup.get(genre_id, "Unknown")
        genre_list.append(GenreModel(id=genre_id, name=genre_name))
    story["genre_list"] = genre_list

    # Get author display name
    if "author_firebase_uid" in story:
        user = get_user_by_firebase_uid(story["author_firebase_uid"])
        if user:
            story["author"] = user.get("display_name", None)
        else:
            story["author"] = None
        del story["author_firebase_uid"]

    return story


def get_story_state(story_id: str) -> StoryStateModel:
    """Get story state from database."""
    state_dict = query_story_state(story_id)
    if state_dict:
        return StoryStateModel(**state_dict)
    return None

def get_story_message(story_id: str) -> list[str]:
    """Get only the stories field from story state."""
    state = get_story_state(story_id)
    if state and state.stories:
        return state.stories
    return []

def write_response_from_prompt(story: str, model: str = "gpt-4o") -> str:
    """Write response from prompt using the workflow.
    
    Args:
        story: The story prompt to generate from
        model: The model to use for generation (default: gpt-4o)
        
    Returns:
        First generated story response
    """
    try:
        # Convert user-friendly model name to actual model name
        model_name = get_model_name(model)
        
        # Create initial state with the input story as user message
        initial_state = StoryStateModel(
            story_id="temp",  # Temporary ID since we don't need to save
            stories=[("user", story)],
            longterm_plots=[],
            guidelines=[],
            requested_act=None,
            conseq_longterm_count=0,
            updated_at=datetime.utcnow()
        )
        
        # Get workflow
        workflow = WorkflowBuilder(
            genre_list=["creative"],  # Default genre since we don't need specific ones
            writer_model=model_name,
        ).compile()
        
        config = {"configurable": {"thread_id": "temp"}}
        
        # Process through workflow
        events = workflow.stream(initial_state.model_dump(), config, stream_mode='values')
        for event in events:
            # Get the first assistant message after our input
            story_messages = event.get("stories", [])
            if len(story_messages) > 1:  # More than our initial message
                role, content = get_message_content(story_messages[-1])
                if role != "user":
                    # Clean up response by removing text after "user:"
                    if "user:" in content.lower():
                        content = content.lower().split("user:")[0]
                    return content.strip()
        
        raise HttpExceptionCustom.internal_server_error
        
    except ValueError as e:
        logger.error(f"Invalid model name: {e}")
        raise HttpExceptionCustom.bad_request
    except Exception as e:
        logger.error(f"Error generating response from prompt: {e}")
        raise HttpExceptionCustom.internal_server_error

async def write_story_message(story_id: str, message: str, model: str = "gpt-4o") -> List[Message]:
    """Process a user message through the LLM workflow and return new messages."""
    logger.info(f"Processing message for story {story_id}")

    try:
        # Convert user-friendly model name to actual model name
        model_name = get_model_name(model)
        
        # Get current state and story
        state = get_story_state(story_id)
        story = query_story(story_id)
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
        workflow = WorkflowBuilder(
            genre_list=genre_names, 
            writer_model=model_name,
        ).compile()
        config = {"configurable": {"thread_id": story_id}}
        
        # Keep track of new messages
        new_messages = []
        last_story_len = len(state.stories)
        
        try:
            output = await workflow.ainvoke(state.model_dump(), config)

            story_messages = output.get("stories", [])
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

            guidelines = output.get("guidelines", [])
            guideline_contents = [g.content for g in guidelines]

            longterm_plots = output.get("longterm_plots", [])
            longterm_plot_contents = [p.content for p in longterm_plots]
            
            # Save final state
            state_model = StoryStateModel(
                story_id=story_id,
                stories=converted_messages,
                longterm_plots=longterm_plot_contents,
                guidelines=guideline_contents,
                requested_act=output.get("requested_act"),
                conseq_longterm_count=output.get("conseq_longterm_count", 0),
                updated_at=datetime.utcnow()
            )
            update_story_state(story_id, state_model)
            
            # Return only the new messages
            return new_messages
            
        except Exception as e:
            logger.error(f"Agent error in workflow: {e}")
            raise HttpExceptionCustom.internal_server_error
        
    except ValueError as e:
        logger.error(f"Invalid model name: {e}")
        raise HttpExceptionCustom.bad_request
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HttpExceptionCustom.internal_server_error

def create_new_story(request: StoryCreateRequestModel) -> str:
    """Create a new story. Also initializes the story state."""
    # Create story document
    story_id = create_story_doc(request)
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
        conseq_longterm_count=0,
        updated_at=datetime.utcnow()
    )

    # Save story state
    story_state_id = create_story_state(story_id, state_model)
    if not story_state_id:
        logger.error(f"Failed to initialize story state for story {story_id}")
        raise HttpExceptionCustom.internal_server_error

    return str(story_id)

def create_story_from_template(request: StoryFromTemplateRequestModel) -> str:
    """Create a new story from template."""
    # Get template
    template = get_template_response(request.template_id)
    if template is None:
        raise HttpExceptionCustom.not_found
    
    # Replace parameters in initial story
    initial_story = template.initial_story
    for param_name, param_value in request.params.items():
        placeholder = "${" + param_name + "}"
        initial_story = initial_story.replace(placeholder, param_value)
    
    # Create story document using StoryCreateRequestModel
    story_request = StoryCreateRequestModel(
        title=template.title,
        description=template.description,
        genre_ids=[genre.id for genre in template.genre_list],
        cover_image=template.cover_image,
        author_firebase_uid=request.author_firebase_uid,
        template_id=request.template_id
    )
    
    story_id = create_story_doc(story_request)
    if not story_id:
        logger.error("Failed to create story document")
        raise HttpExceptionCustom.internal_server_error
    
    # Initialize story state with the processed initial story
    state_model = StoryStateModel(
        story_id=str(story_id),
        stories=[("user", initial_story)],
        longterm_plots=[],
        guidelines=[],
        requested_act=None,
        conseq_longterm_count=0,
        updated_at=datetime.utcnow()
    )
    
    # Save story state
    story_state_id = create_story_state(story_id, state_model)
    if not story_state_id:
        logger.error(f"Failed to initialize story state for story {story_id}")
        raise HttpExceptionCustom.internal_server_error
    
    return str(story_id)

def delete_story_response(story_id: str) -> bool:
    """Delete a story and its state."""
    logger.info(f"Deleting story {story_id}")
    
    # Check if story exists
    story = query_story(story_id)
    if not story:
        raise HttpExceptionCustom.not_found("Story not found")
    
    # Delete story and its state
    success = delete_story(story_id)
    if not success:
        raise HttpExceptionCustom.internal_server_error("Failed to delete story")
    
    return True

def update_story_response(story_id: str, request: StoryUpdateRequestModel) -> bool:
    """Update story details."""
    logger.info(f"Updating story {story_id}")
    
    # Check if story exists
    story = query_story(story_id)
    if not story:
        raise HttpExceptionCustom.not_found("Story not found")
    
    # Convert request model to dict and remove None values
    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    
    if not update_data:
        logger.warning("No fields to update")
        return True
    
    # Update story
    success = update_story(story_id, update_data)
    if not success:
        raise HttpExceptionCustom.internal_server_error("Failed to update story")
    
    return True

def edit_story_messages(story_id: str, message_updates: List[MessageEditItem]) -> List[Message]:
    """Edit specific messages in a story's message history.
    
    Args:
        story_id: The ID of the story to edit
        message_updates: List of message edit operations
        
    Returns:
        Updated list of messages
    """
    # Get current state
    state = get_story_state(story_id)
    if not state:
        logger.error(f"No state found for story {story_id}")
        raise HttpExceptionCustom.not_found("Story not found")
    
    # Sort operations by index in reverse order to handle deletes and inserts properly
    message_updates.sort(key=lambda x: x.index, reverse=True)
    
    # Process each operation
    for update in message_updates:
        # Validate index
        if update.index < 0:
            raise HttpExceptionCustom.bad_request(f"Invalid message index: {update.index}")
            
        if update.operation == MessageOperation.EDIT:
            if update.index >= len(state.stories):
                raise HttpExceptionCustom.bad_request(f"Invalid message index for edit: {update.index}")
            if not update.content:
                raise HttpExceptionCustom.bad_request("Content required for edit operation")
            # Preserve original role
            original_role = state.stories[update.index][0]
            state.stories[update.index] = (original_role, update.content)
            
        elif update.operation == MessageOperation.DELETE:
            if update.index >= len(state.stories):
                raise HttpExceptionCustom.bad_request(f"Invalid message index for delete: {update.index}")
            state.stories.pop(update.index)
            
        elif update.operation == MessageOperation.INSERT:
            if update.index > len(state.stories):
                raise HttpExceptionCustom.bad_request(f"Invalid message index for insert: {update.index}")
            if not update.content:
                raise HttpExceptionCustom.bad_request("Content required for insert operation")
            state.stories.insert(update.index, (update.role, update.content))
    
    # Update state in database
    state.updated_at = datetime.utcnow()
    success = update_story_state(story_id, state)
    if not success:
        raise HttpExceptionCustom.internal_server_error("Failed to update story state")
    
    return state.stories
 