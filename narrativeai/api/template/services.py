from datetime import datetime
from typing import List, Optional, Dict
import re

from .database import create_template, get_template, list_templates
from .schema import TemplateCreateRequestModel, TemplateModel, TemplateListItemModel
from ..genre.database import query_list_genres
from ..genre.schema import GenreModel
from ..user.database import get_user_by_firebase_uid

def extract_params_from_story(story: str) -> Dict[str, str]:
    """Extract parameters from story text using ${param} syntax."""
    param_pattern = r'\${([^}]+)}'
    params = {}
    matches = re.finditer(param_pattern, story)
    
    for match in matches:
        param_name = match.group(1)
        # Set default value as empty string
        params[param_name] = ""
        
    return params

def create_new_template(request: TemplateCreateRequestModel) -> str:
    """Create a new template and return its ID."""
    # Extract parameters from initial story if not provided
    if not request.params:
        params = extract_params_from_story(request.initial_story)
    else:
        params = request.params
    
    # Create template
    template_id = create_template(
        title=request.title,
        description=request.description,
        genre_list=request.genre_ids,  # Store just IDs in database
        initial_story=request.initial_story,
        params=params,
        cover_image=request.cover_image,
        author_firebase_uid=request.author_firebase_uid,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    return template_id

def get_template_response(template_id: str) -> Optional[TemplateModel]:
    """Get template by ID."""
    template = get_template(template_id)
    if template is None:
        return None

    # Create a lookup dictionary for faster genre name retrieval
    genre_lists = query_list_genres()
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_lists}
    
    # Process each story
    genre_list = []
    for genre_id in template["genre_list"]:
        genre_name = genre_lookup.get(genre_id, "Unknown")
        genre_list.append(GenreModel(id=genre_id, name=genre_name))
    template["genre_list"] = genre_list
    
    # Get author display name
    if "author_firebase_uid" in template:
        user = get_user_by_firebase_uid(template["author_firebase_uid"])
        if user:
            template["author"] = user.get("display_name", None)
        else:
            template["author"] = None
        del template["author_firebase_uid"]
    
    return TemplateModel(**template)

def list_templates_response(
    skip: int = 0,
    limit: int = 10,
    author_firebase_uid: Optional[str] = None
) -> List[TemplateListItemModel]:
    """List templates with optional author filter, returning only essential fields."""
    templates = list_templates(skip, limit, author_firebase_uid)
    
    # Create a lookup dictionary for faster genre name retrieval
    genre_lists = query_list_genres()
    genre_lookup = {genre["id"]: genre["name"] for genre in genre_lists}
    
    # Process each template
    simplified_templates = []
    for template in templates:
        # Process genre information
        genre_list = []
        for genre_id in template["genre_list"]:
            genre_name = genre_lookup.get(genre_id, "Unknown")
            genre_list.append(GenreModel(id=genre_id, name=genre_name))
        
        # Get author display name
        author = None
        if "author_firebase_uid" in template:
            user = get_user_by_firebase_uid(template["author_firebase_uid"])
            if user:
                author = user.get("display_name", None)
        
        # Create simplified template item
        simplified_template = TemplateListItemModel(
            id=template["id"],
            title=template["title"],
            description=template["description"],
            genre_list=genre_list,
            cover_image=template.get("cover_image"),
            author=author
        )
        simplified_templates.append(simplified_template)
    
    return simplified_templates 