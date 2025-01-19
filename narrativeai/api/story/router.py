from fastapi import APIRouter, Depends
import logging

from ..dependencies import GenericOKResponse, common_pagination_parameters, HttpException
from .schema import ListStoryResponseModel, StoryModel
from .services import list_stories_response

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/story",
    tags=["story"],
    dependencies=[
        Depends(common_pagination_parameters),
        Depends(GenericOKResponse),
        Depends(HttpException),
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
