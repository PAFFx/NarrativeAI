
from fastapi import APIRouter, Depends, HTTPException
import logging

from ..dependencies import GenericOKResponse, common_pagination_parameters, HttpExceptionCustom
from .schema import GenreModel, ListGenreResponseModel
from .database import query_list_genres

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/genre",
    tags=["genre"],
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
        response_model=ListGenreResponseModel,
        )
async def get_genre_list():
    genres: list[GenreModel] = query_list_genres()
    return ListGenreResponseModel(genres=genres)
