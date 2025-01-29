from fastapi import APIRouter, Depends, HTTPException
import logging

from ..dependencies import GenericOKResponse, common_pagination_parameters, HttpExceptionCustom
from .schema import ListStoryResponseModel, StoryCreateRequestModel, StoryMessageModel, StoryModel, StoryStateModel, WriteRequestModel, StoryCreateResponseModel
from .services import create_new_story, get_story_message, list_stories_response, write_story_message, get_story_response

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

@router.get(
        "/",
        status_code=200,
        response_model_exclude_none=True,
        response_model=ListStoryResponseModel,
        )
async def get_story_list(skip: int = 0, limit: int = 30):
    stories: list[StoryModel] = list_stories_response(skip, limit)
    return ListStoryResponseModel(stories=stories)

@router.post(
        "/",
        status_code=201,
        response_model_exclude_none=True,
        response_model=StoryCreateResponseModel,
)
async def post_story(request: StoryCreateRequestModel):
    """Create a new story."""
    try:
        story_id = await create_new_story(request)
        return StoryCreateResponseModel(story_id=story_id)
    except Exception as e:
        logger.error(f"Error creating story: {e}")
        raise HttpExceptionCustom.internal_server_error


@router.get("/{story_id}", response_model=StoryModel)
async def get_story(story_id: str):
    story = await get_story_response(story_id)
    return story


@router.get("/{story_id}/messages", response_model=StoryMessageModel)
async def get_story_messages(story_id: str):
    """Get story messages from database."""
    messages = await get_story_message(story_id)
    return StoryMessageModel(messages=messages)

@router.post("/{story_id}/write", response_model=StoryMessageModel)
async def write_story(story_id: str, request: WriteRequestModel):
    """Write to story and process through LLM workflow."""
    new_messages = await write_story_message(story_id, request.message)
    if new_messages is None:
        raise HttpExceptionCustom.not_found
    
    return StoryMessageModel(messages=new_messages)
