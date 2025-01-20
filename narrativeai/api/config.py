from starlette.config import Config

config = Config(".env") #load env

MONGODB_URI = config(
    "MONGODB_URI",
    default="mongodb://root:verystrongrootpassword@localhost:27017",
    )

DB_NAME = config(
    "MONGODB_NAME",
    default="NarrativeAI",
    )

#Collection names
DB_STORY_COLLECTION = "Story"
DB_GENRE_COLLECTION = "Genre"
DB_STORY_STATE_COLLECTION = "StoryState"



    

