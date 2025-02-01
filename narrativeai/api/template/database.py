from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId
from ..database import db_client

# Get templates collection from db_client
templates_collection = db_client.template_collection 

def create_template(
    title: str,
    description: str,
    genre_list: List[str],
    initial_story: str,
    params: Dict[str, str],
    cover_image: Optional[str],
    author_firebase_uid: str,
    created_at: datetime,
    updated_at: datetime
) -> str:
    """Create a new template in the database."""
    result = templates_collection.insert_one({
        "title": title,
        "description": description,
        "genre_list": genre_list,  # Store as strings, not ObjectIds
        "initial_story": initial_story,
        "params": params,
        "cover_image": cover_image,
        "author_firebase_uid": author_firebase_uid,
        "created_at": created_at,
        "updated_at": updated_at
    })
    return str(result.inserted_id)

def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Get template by ID."""
    template = templates_collection.find_one({"_id": ObjectId(template_id)})
    if template:
        template["id"] = str(template.pop("_id"))
        # Ensure genre_list contains strings
        template["genre_list"] = [str(genre_id) for genre_id in template["genre_list"]]
    return template

def list_templates(
    skip: int = 0,
    limit: int = 10,
    author_firebase_uid: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List templates with optional author filter."""
    query = {}
    if author_firebase_uid:
        query["author_firebase_uid"] = author_firebase_uid
        
    cursor = templates_collection.find(query).skip(skip).limit(limit)
    templates = []
    for template in cursor:
        template["id"] = str(template.pop("_id"))
        # Ensure genre_list contains strings
        template["genre_list"] = [str(genre_id) for genre_id in template["genre_list"]]
        templates.append(template)
    return templates 