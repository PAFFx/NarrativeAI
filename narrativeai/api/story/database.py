import logging
from bson.objectid import ObjectId
from bson.errors import InvalidId

from ..database import db_client
from ..dependencies import HttpException

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
        raise HttpException.bad_request

def query_list_genre() -> list:
    try:
        genres_cursor = db_client.genre_collection.find({})
        genres = []
        for genre in genres_cursor:
            genre["id"] = str(genre["_id"])
            genres.append(genre)
        return genres

    except InvalidId:
        raise HttpException.bad_request
