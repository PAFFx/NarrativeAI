from .database import query_list_stories, query_list_genre
from .schema import StoryModel, GenreModel
import logging

logger = logging.getLogger(__name__)

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


    return stories
