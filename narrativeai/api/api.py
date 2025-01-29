import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .database import db_client

from .story.router import router as story_router
from .genre.router import router as genre_router
from .user.router import router as user_router

app = FastAPI()
# add router here
app.include_router(story_router)
app.include_router(genre_router)
app.include_router(user_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    db_client.list_database_names()
except:
    sys.exit("Error: Database connection failed")


@app.get("/", status_code=200)
def read_root():
    return {"Hello": "World"}


def start_dev_server():
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run("narrativeai.api.api:app", reload=True ,log_config=log_config)