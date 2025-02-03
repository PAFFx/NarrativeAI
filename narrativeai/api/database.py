from pymongo import MongoClient
from .config import (
    DB_STORY_STATE_COLLECTION,
    MONGODB_URI,
    DB_NAME,
    DB_STORY_COLLECTION,
    DB_GENRE_COLLECTION,
    DB_USER_COLLECTION,
    DB_TEMPLATE_COLLECTION,
)

class DB_client(MongoClient):
    def __init__(self, *args, **kwargs):
        super(DB_client, self).__init__(*args, **kwargs)
        self.db = self.get_database(DB_NAME)

        #collections
        self.story_collection = self.db.get_collection(DB_STORY_COLLECTION)
        self.genre_collection = self.db.get_collection(DB_GENRE_COLLECTION)
        self.story_states_collection = self.db.get_collection(DB_STORY_STATE_COLLECTION)
        self.user_collection = self.db.get_collection(DB_USER_COLLECTION)
        self.template_collection = self.db.get_collection(DB_TEMPLATE_COLLECTION)

db_client = DB_client(MONGODB_URI)