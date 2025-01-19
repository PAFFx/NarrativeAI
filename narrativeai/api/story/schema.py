from pydantic import BaseModel

class GenreModel(BaseModel):
    id: str
    name: str

class StoryModel(BaseModel):
    id: str
    title: str
    description: str
    genre_list: list[GenreModel]
    cover_image: str | None = None
    author: str | None = None

class ListStoryResponseModel(BaseModel):
    stories: list[StoryModel]

