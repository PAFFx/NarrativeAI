from pydantic import BaseModel
from typing import List, Optional

class GenreModel(BaseModel):
    id: str
    name: str

class ListGenreResponseModel(BaseModel):
    genres: list[GenreModel]
