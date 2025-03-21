from fastapi import APIRouter, Depends, HTTPException
import logging

from ..dependencies import GenericOKResponse, common_pagination_parameters, HttpExceptionCustom
from .schema import (
    ListStoryResponseModel,
    StoryCreateRequestModel,
    StoryMessageModel,
    StoryModel,
    StoryStateModel,
    WriteFromPromptRequestModel,
    WriteFromPromptResponseModel,
    WriteRequestModel,
    StoryCreateResponseModel,
    StoryFromTemplateRequestModel,
    StoryUpdateRequestModel,
    MessageEditRequestModel
)
from .services import (
    create_new_story,
    get_story_message,
    list_stories_response,
    write_response_from_prompt,
    write_story_message,
    get_story_response,
    create_story_from_template,
    delete_story_response,
    update_story_response,
    edit_story_messages
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/story",
    tags=["story"],
    dependencies=[
        Depends(common_pagination_parameters),
        Depends(GenericOKResponse),
        Depends(HttpExceptionCustom),
    ],
)

@router.get("/", response_model=ListStoryResponseModel, response_model_exclude_none=True)
def list_stories(
    skip: int = 0,
    limit: int = 10,
    author_firebase_uid: str | None = None
):
    """List stories with optional author filter."""
    stories = list_stories_response(skip, limit, author_firebase_uid)
    return ListStoryResponseModel(stories=stories)

@router.post(
        "/",
        status_code=201,
        response_model_exclude_none=True,
        response_model=StoryCreateResponseModel,
)
def post_story(request: StoryCreateRequestModel):
    """Create a new story."""
    try:
        story_id = create_new_story(request)
        return StoryCreateResponseModel(story_id=story_id)
    except Exception as e:
        logger.error(f"Error creating story: {e}")
        raise HttpExceptionCustom.internal_server_error


@router.get("/{story_id}", response_model=StoryModel)
def get_story(story_id: str):
    story = get_story_response(story_id)
    return story


@router.get("/{story_id}/messages", response_model=StoryMessageModel)
def get_story_messages(story_id: str):
    """Get story messages from database."""
    messages = get_story_message(story_id)
    return StoryMessageModel(messages=messages)

@router.post("/{story_id}/write", response_model=StoryMessageModel)
async def write_story(story_id: str, request: WriteRequestModel):
    """Write to story and process through LLM workflow."""
    if request.model is None:
        new_messages = await write_story_message(story_id, request.message)
    else:
        new_messages = await write_story_message(story_id, request.message, request.model)
    if new_messages is None:
        raise HttpExceptionCustom.not_found
    
    return StoryMessageModel(messages=new_messages)

@router.post("/write_from_prompt", response_model=WriteFromPromptResponseModel)
def write_from_prompt(request: WriteFromPromptRequestModel):
    """Write from prompt."""
    story = request.story
    model = request.model
    new_message = write_response_from_prompt(story, model)
    return WriteFromPromptResponseModel(next_story=new_message)

@router.post(
    "/from_template",
    status_code=201,
    response_model=StoryCreateResponseModel,
)
def create_from_template(request: StoryFromTemplateRequestModel):
    """Create a new story from template."""
    try:
        logger.info(f"Creating story from template: {request}")
        story_id = create_story_from_template(request)
        return StoryCreateResponseModel(story_id=story_id)
    except Exception as e:
        logger.error(f"Error creating story from template: {e}")
        raise HttpExceptionCustom.internal_server_error

@router.delete("/{story_id}", response_model=bool)
def delete_story(story_id: str):
    """Delete a story and its state."""
    try:
        return delete_story_response(story_id)
    except Exception as e:
        logger.error(f"Error deleting story: {e}")
        raise HttpExceptionCustom.internal_server_error

@router.patch("/{story_id}", response_model=bool)
def update_story(story_id: str, request: StoryUpdateRequestModel):
    """Update story details."""
    try:
        return update_story_response(story_id, request)
    except Exception as e:
        logger.error(f"Error updating story: {e}")
        raise HttpExceptionCustom.internal_server_error

@router.patch("/{story_id}/messages", response_model=StoryMessageModel)
def update_story_messages(story_id: str, request: MessageEditRequestModel):
    """Update specific messages in a story."""
    try:
        updated_messages = edit_story_messages(story_id, request.messages)
        return StoryMessageModel(messages=updated_messages)
    except Exception as e:
        logger.error(f"Error updating story messages: {e}")
        raise HttpExceptionCustom.internal_server_error
