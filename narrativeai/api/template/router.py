from fastapi import APIRouter, Depends, HTTPException
import logging

from ..dependencies import GenericOKResponse, common_pagination_parameters, HttpExceptionCustom
from .schema import ListTemplateResponseModel, TemplateCreateRequestModel, TemplateModel, TemplateCreateResponseModel
from .services import create_new_template, get_template_response, list_templates_response

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/template",
    tags=["template"],
    dependencies=[
        Depends(common_pagination_parameters),
        Depends(GenericOKResponse),
        Depends(HttpExceptionCustom),
    ],
)

@router.get("", response_model=ListTemplateResponseModel)
def list_templates(
    skip: int = 0,
    limit: int = 10,
    author_firebase_uid: str | None = None
):
    """List templates with optional author filter."""
    templates = list_templates_response(skip, limit, author_firebase_uid)
    return ListTemplateResponseModel(templates=templates)

@router.post(
    "",
    status_code=201,
    response_model_exclude_none=True,
    response_model=TemplateCreateResponseModel,
)
def post_template(request: TemplateCreateRequestModel):
    """Create a new template."""
    try:
        template_id = create_new_template(request)
        return TemplateCreateResponseModel(template_id=template_id)
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HttpExceptionCustom.internal_server_error

@router.get("/{template_id}", response_model=TemplateModel)
def get_template(template_id: str):
    """Get template by ID."""
    template = get_template_response(template_id)
    if template is None:
        raise HttpExceptionCustom.not_found
    return template 