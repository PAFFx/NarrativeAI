from datetime import datetime, timezone
from pydantic import BaseModel
from fastapi import HTTPException
from datetime import datetime, timezone
import mimetypes
import httpx


def common_pagination_parameters(skip: int = 0, limit: int = 10000):
    return {"skip": skip, "limit": limit}


class GenericOKResponse(BaseModel):
    detail: str = "OK"


class HttpException:
    # 4xx
    bad_request = HTTPException(400, detail="Bad Request")
    unauthorized = HTTPException(401, detail="Unauthorized")
    forbidden = HTTPException(
        403, detail="Forbidden"
    )  # unlike unauthorized the client's identity is known.
    not_found = HTTPException(404, detail="Not Found")
    method_not_allowed = HTTPException(405, detail="Method Not Allowed")
    request_timeout = HTTPException(408, detail="Request Timeout")
    unprocessable_content = HTTPException(422, detail="Unprocessable Content")

    # 5xx
    internal_server_error = HTTPException(500, detail="Internal Server Error")

    # MEME
    teapot = HTTPException(418, detail="I'm a teapot")


# TYPE
student_type = "student"
teacher_type = "teacher"
course_type = "course"
class_type = "class"
reply_type_assignment = "reply_assignment"
reply_type_thread = "reply_thread"


# FIX: Doesn't work for youtube video
def CheckHttpFileType(url: str) -> str:
    result = mimetypes.guess_type(url)[0]  # ('audio/mpeg', None)
    if result == None:  # no extension url
        # download only 'Content-Type' metadata
        response = httpx.head(url).headers[
            "Content-Type"
        ]  # ex. 'texts/html; charset=utf-8' or 'application/pdf'
        result = response.split(";")[0]
    file_type, extension = result.split("/")  # 'video/mp4'
    return file_type


def mongo_datetime_to_timestamp(dt: datetime) -> int:
    """
    Converts a datetime object to a Unix timestamp.

    Args:
        dt (datetime): The datetime object to convert.

    Returns:
        int: The Unix timestamp (in UTC) equivalent of the datetime object.
    """
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def timestamp_to_utc_datetime(timestamp: int) -> datetime:
    """
    Converts a Unix timestamp to a datetime object.

    Args:
        timestamp (int): The Unix timestamp (in UTC) to convert.

    Returns:
        datetime: The datetime object (in UTC) equivalent of the Unix timestamp.
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def utc_datetime(dt: datetime) -> datetime:
    """
    Adds utc timezone to a datetime object.

    Args:
        dt (datetime): The datetime object to add timezone to.

    Returns:
        datetime: The datetime object with timezone.
    """
    return dt.replace(tzinfo=timezone.utc)


def utc_datetime_now() -> datetime:
    """
    Gets the current datetime in UTC.

    Returns:
        datetime: The current datetime in UTC.
    """
    return datetime.now(tz=timezone.utc)


def get_timestamp_from_datetime(dt: datetime) -> int:
    return int(datetime.replace(dt, tzinfo=timezone.utc).timestamp())