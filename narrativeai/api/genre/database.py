import logging
from datetime import datetime
from bson.objectid import ObjectId
from bson.errors import InvalidId

from ..database import db_client
from ..dependencies import HttpExceptionCustom

logger = logging.getLogger(__name__)

def query_list_genres():
    try:
        genres_cursor = db_client.genre_collection.find()
        genres = []
        for genre in genres_cursor:
            genre["id"] = str(genre["_id"])
            genres.append(genre)
        return genres
    except Exception as e:
        logger.error(f"Error querying genres: {e}")
        raise HttpExceptionCustom.internal_server_error
