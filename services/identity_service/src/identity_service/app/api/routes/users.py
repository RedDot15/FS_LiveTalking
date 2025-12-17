import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from postgresql_client.model.models import UserModel

from authorization import (
    CurrentToken,
    has_authority,
    TokenPayload
)

from identity_service.app.core.config import settings
from identity_service.app.core.security import get_password_hash, verify_password
from identity_service.app.models import (
    Message,
    UpdatePassword,
    UserCreate,
    UserPublic,
    UserRegister,
    UsersPublic,
    UserUpdate,
    UserUpdateMe
)
from identity_service.app.utils import (
    generate_new_account_email, 
    send_email
)

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/signup", response_model=UserPublic)
def register_user(request: Request, user_in: UserRegister) -> Any:
    """
    Create new user without the need to be logged in.
    """

    with request.app.state.postgres.get_session() as session:
        user = request.app.state.postgres.get_user_by_email(session=session, email=user_in.email)
        if user:
            raise HTTPException(
                status_code=400,
                detail="The user with this email already exists in the system.",
            )

        db_obj = UserModel()

        user_data = user_in.model_dump(
            exclude_none=True, exclude={"password"}
        )

        for key, value in user_data.items():
            if value is not None:
                setattr(db_obj, key, value)

        setattr(db_obj, "id", uuid.uuid4())
        setattr(db_obj, "password", get_password_hash(user_in.password))

        default_role_id = request.app.state.postgres.get_role_by_name(session=session, name="USER").id

        db_obj = request.app.state.postgres.insert_user(
            session=session, db_obj=db_obj, role_ids=[str(default_role_id)]
        )

        if settings.emails_enabled and user_in.email:
            email_data = generate_new_account_email(
                email_to=user_in.email, username=user_in.email, password=user_in.password
            )
            send_email(
                email_to=user_in.email,
                subject=email_data.subject,
                html_content=email_data.html_content,
            )

        return UserPublic.model_validate(db_obj)

@router.get("/me", response_model=UserPublic)
def read_user_me(request: Request, current_token: CurrentToken) -> Any:
    """
    Get current user.
    """
    with request.app.state.postgres.get_session() as session:
        return UserPublic.model_validate(request.app.state.postgres.get_user_by_id(session, current_token.id))

@router.put("/me", response_model=UserPublic)
def update_user_me(
    *, request: Request, user_in: UserUpdateMe, current_token: CurrentToken
) -> Any:
    """
    Update own user.
    """
    with request.app.state.postgres.get_session() as session:
        if user_in.email:
            existing_user = request.app.state.postgres.get_user_by_email(session=session, email=user_in.email)
            if existing_user and existing_user.id != current_token.id:
                raise HTTPException(
                    status_code=409, detail="User with this email already exists"
                )

        db_obj = request.app.state.postgres.get_user_by_id(session, current_token.id)

        user_data = user_in.model_dump(exclude_none=True)

        for key, value in user_data.items():
            if value is not None:
                setattr(db_obj, key, value)

        return request.app.state.postgres.update_user(
            session=session,
            db_obj=db_obj,
        )

@router.put("/me/password", response_model=Message)
def update_password_me(
    *, request: Request, body: UpdatePassword, current_token: CurrentToken
) -> Any:
    """
    Update own password.
    """
    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=400, detail="New password cannot be the same as the current one"
        )

    with request.app.state.postgres.get_session() as session:
        db_obj = request.app.state.postgres.get_user_by_id(session, current_token.id)

        if not verify_password(body.current_password, db_obj.password):
            raise HTTPException(status_code=400, detail="Incorrect password")

        db_obj.password = get_password_hash(body.new_password)

        request.app.state.postgres.update_user(session=session, db_obj=db_obj)

    return Message(message="Password updated successfully")

# @router.delete("/me", response_model=Message)
# def delete_user_me(request: Request, current_token: CurrentToken) -> Any:
#     """
#     Delete own user.
#     """
#     with request.app.state.postgres.get_session() as session:
#         request.app.state.postgres.delete_user(session, current_token.id)

#     return Message(message="User deleted successfully")

@router.post(
    "", 
    dependencies=[Depends(has_authority(authority="CREATE_USER"))], 
    response_model=UserPublic
)
def create_user(*, request: Request, user_in: UserCreate) -> Any:
    """
    Create new user.
    """
    db_obj: UserModel = None
    with request.app.state.postgres.get_session() as session:
        user = request.app.state.postgres.get_user_by_email(session=session, email=user_in.email)
        if user:
            raise HTTPException(
                status_code=400,
                detail="The user with this email already exists in the system.",
            )
        user = request.app.state.postgres.get_user_by_username(session=session, username=user_in.username)
        if user:
            raise HTTPException(
                status_code=400,
                detail="The user with this email already exists in the system.",
            )
        
        db_obj = UserModel()

        db_obj.id=uuid.uuid4()
        db_obj.username=user_in.username
        db_obj.password=get_password_hash(user_in.password)
        db_obj.name=user_in.name
        db_obj.avatar_url=user_in.avatar_url
        db_obj.email=user_in.email
        db_obj.phone_number=user_in.phone_number
    
        db_obj = request.app.state.postgres.insert_user(
            session=session, db_obj=db_obj, role_ids=user_in.role_ids
        )

        # Convert UserModel -> UserPublic
        user_public = UserPublic.model_validate(db_obj)

        return user_public

@router.get(
    "",
    dependencies=[Depends(has_authority(authority="READ_USERS"))], 
    response_model=UsersPublic,
)
def read_users(request: Request, skip: int = 0, limit: int = 100) -> Any:
    """
    Retrieve users.
    """
    with request.app.state.postgres.get_session() as session:
        user_public_list = []
        for user_model in request.app.state.postgres.get_users(
            session=session, offset=skip, limit=limit
        ):
            user_public_list.append(UserPublic.model_validate(user_model))
        return UsersPublic(data=user_public_list)

@router.get(
    "/{user_id}", 
    dependencies=[Depends(has_authority(authority="READ_USER"))], 
    response_model=UserPublic)
def read_user_by_id(
    request: Request, user_id: uuid.UUID
) -> Any:
    """
    Get a specific user by id.
    """
    with request.app.state.postgres.get_session() as session:
        user_model = request.app.state.postgres.get_user_by_id(session, user_id)
        return UserPublic.model_validate(user_model)

@router.put(
    "/{user_id}",
    response_model=UserPublic,
)
def update_user(
    *,
    request: Request,
    user_id: uuid.UUID,
    user_in: UserUpdate,
    permitted_token: Annotated[TokenPayload, Depends(has_authority(authority="UPDATE_USER"))]
) -> Any:
    """
    Update a user.
    """
    if user_id == permitted_token.id:
        raise HTTPException(
                status_code=403,
                detail="Cannot perform self-update with this api.",
            )

    with request.app.state.postgres.get_session() as session:
        db_user = request.app.state.postgres.get_user_by_id(session, user_id)
        if not db_user:
            raise HTTPException(
                status_code=404,
                detail="The user with this id does not exist in the system",
            )
        for user_role in db_user.user_roles:
            if user_role.role.name == "ADMIN":
                raise HTTPException(
                    status_code=403,
                    detail="Cannot perform update with an user whose role is ADMIN",
                )
        if user_in.email:
            existing_user = request.app.state.postgres.get_user_by_email(session=session, email=user_in.email)
            if existing_user and existing_user.id != permitted_token.id:
                raise HTTPException(
                    status_code=409, detail="User with this email already exists"
                )

        user_data = user_in.model_dump(
            exclude_none=True, exclude={"role_ids", "password"}
        )

        for key, value in user_data.items():
            if value is not None:
                setattr(db_user, key, value)

        if "password" in user_data:
            hashed_password = get_password_hash(user_data["password"])
            db_user.password = hashed_password

        db_user = request.app.state.postgres.update_user(
            session=session, db_obj=db_user, role_ids=user_in.role_ids
        )

        return UserPublic.model_validate(db_user)

@router.delete("/{user_id}")
async def delete_user(request: Request, user_id: uuid.UUID, permitted_token: Annotated[TokenPayload, Depends(has_authority(authority="DELETE_USER"))]) -> Message:
    """
    Delete a user.
    """
    # Cannot self delete
    if user_id == permitted_token.id:
        raise HTTPException(
                status_code=403,
                detail="Cannot perform self-update with this api.",
            )
    with request.app.state.postgres.get_session() as session:
        # Cannot delete user have role ADMIN
        db_user = request.app.state.postgres.get_user_by_id(session, user_id)
        for user_role in db_user.user_roles:
            if user_role.role.name == "ADMIN":
                raise HTTPException(
                    status_code=403,
                    detail="Cannot perform update with an user whose role is ADMIN",
                )

        # Delete user
        request.app.state.postgres.delete_user(session, user_id)

    await request_delete_datas(user_id=user_id, request=request)

    return Message(message="User deleted successfully")

