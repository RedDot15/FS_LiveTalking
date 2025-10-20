import uuid
from typing import Annotated, Any

from authorization.model import TokenPayload
from fastapi import APIRouter, Depends, HTTPException, Request
from postgresql_client.model.models import RoleModel

from authorization import (
    has_authority
)

from identity_service.app.core.security import get_password_hash
from identity_service.app.models import (
    Message,
    RoleCreate,
    RolePublic,
    RolesPublic,
    RoleUpdate
)

router = APIRouter(prefix="/roles", tags=["roles"])

@router.post(
    "", 
    dependencies=[Depends(has_authority(authority="CREATE_ROLE"))], 
    response_model=RolePublic
)
def create_role(*, request: Request, role_in: RoleCreate) -> Any:
    """
    Create new role.
    """
    if not role_in.permission_ids:
        raise HTTPException(
                    status_code=400,
                    detail="Permission_ids is required.",
                )

    db_obj: RoleModel = None
    with request.app.state.postgres.get_session() as session:
        role = request.app.state.postgres.get_role_by_name(session=session, email=role_in.name)
        if role:
            raise HTTPException(
                status_code=400,
                detail="The role with this name already exists in the system.",
            )

        db_obj = RoleModel()

        db_obj.id=uuid.uuid4()
        db_obj.name=role_in.name
    
        db_obj = request.app.state.postgres.insert_role(
            session=session, db_obj=db_obj, role_ids=role_in.permission_ids
        )

        # Convert RoleModel -> RolePublic
        role_public = RolePublic.model_validate(db_obj)

        return role_public

@router.get(
    "",
    dependencies=[Depends(has_authority(authority="READ_ROLES"))], 
    response_model=RolesPublic,
)
def read_roles(request: Request, skip: int = 0, limit: int = 100) -> Any:
    """
    Retrieve roles.
    """
    with request.app.state.postgres.get_session() as session:
        role_public_list = []
        for role_model in request.app.state.postgres.get_roles(
            session=session, offset=skip, limit=limit
        ):
            role_public_list.append(RolePublic.model_validate(role_model))
        return RolesPublic(data=role_public_list)

@router.get(
    "/{role_id}", 
    dependencies=[Depends(has_authority(authority="READ_ROLE"))], 
    response_model=RolePublic)
def read_role_by_id(
    request: Request, role_id: uuid.UUID
) -> Any:
    """
    Get a specific role by id.
    """
    with request.app.state.postgres.get_session() as session:
        role_model = request.app.state.postgres.get_role_by_id(session, role_id)
        return RolePublic.model_validate(role_model)

@router.put(
    "/{role_id}",
    response_model=RolePublic,
)
def update_role(
    *,
    request: Request,
    role_id: uuid.UUID,
    role_in: RoleUpdate,
    permitted_token: Annotated[TokenPayload, Depends(has_authority(authority="UPDATE_ROLE"))]
) -> Any:
    """
    Update a role.
    """
    if not role_in.permission_ids:
        raise HTTPException(
                    status_code=400,
                    detail="Role_ids is required.",
                )
    
    with request.app.state.postgres.get_session() as session:
        db_role = request.app.state.postgres.get_role_by_id(session, role_id)
        if not db_role:
            raise HTTPException(
                status_code=404,
                detail="The role with this id does not exist in the system",
            )
        existing_role = request.app.state.postgres.get_role_by_name(session=session, name=role_in.name)
        if existing_role and existing_role.id != permitted_token.id:
            raise HTTPException(
                status_code=409, detail="Role with this name already exists"
            )

        role_data = role_in.model_dump(
            exclude_none=True, exclude={"permission_ids"}
        )

        for key, value in role_data.items():
            if value is not None:
                setattr(db_role, key, value)

        db_role = request.app.state.postgres.update_role(
            session=session, db_obj=db_role, permission_ids=role_in.permission_ids
        )

        return RolePublic.model_validate(db_role)

@router.delete("/{role_id}", dependencies=[Depends(has_authority(authority="DELETE_ROLE"))])
def delete_role(request: Request, role_id: uuid.UUID) -> Message:
    """
    Delete a role.
    """
    with request.app.state.postgres.get_session() as session:
        request.app.state.postgres.delete_role(session, role_id)

    return Message(message="Role deleted successfully")