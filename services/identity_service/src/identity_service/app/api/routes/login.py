from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from authorization import TokenPayload

from ...core.config import settings
from ...core.security import create_access_token, get_password_hash, verify_password, verify_token
from ....app.models import Message, NewPassword, Token, TokenPayload
from ....app.utils import (
    generate_password_reset_token,
    generate_reset_password_email,
    send_email,
    verify_password_reset_token,
)

router = APIRouter(tags=["login"])

@router.post("/login/access-token")
def login_access_token(
    request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    with request.app.state.postgres.get_session() as session:
        db_user = request.app.state.postgres.get_user_by_username(
            session=session,
            username=form_data.username,
        )
        if not db_user or not verify_password(form_data.password, db_user.password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        return Token(
            access_token=create_access_token(
                db_user ,expires_delta=access_token_expires
            )
        )

@router.post("/login/test-token", response_model=TokenPayload)
def test_token(token: Token) -> Any:
    """
    Test access token
    """
    print("Token received for testing:", token)  
    return TokenPayload(**verify_token(token=token.access_token))

@router.post("/password-recovery/{email}")
def recover_password(request: Request, email: str) -> Message:
    """
    Password Recovery
    """
    with request.app.state.postgres.get_session() as session:
        user = request.app.state.postgres.get_user_by_email(session=session, email=email)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="The user with this email does not exist in the system.",
            )

        password_reset_token = generate_password_reset_token(email=email)
        email_data = generate_reset_password_email(
            email_to=user.email, email=email, token=password_reset_token
        )
        send_email(
            email_to=user.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )

        return Message(message="Password recovery email sent")


@router.post("/reset-password")
def reset_password(request: Request, body: NewPassword) -> Message:
    """
    Reset password
    """
    email = verify_password_reset_token(token=body.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token")
    with request.app.state.postgres.get_session() as session:
        user = request.app.state.postgres.get_user_by_email(session=session, email=email)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="The user with this email does not exist in the system.",
            )
        hashed_password = get_password_hash(password=body.new_password)
        user.password = hashed_password
        request.app.state.postgres.update_user(session=session, db_obj=user)
        return Message(message="Password updated successfully")


# @router.post(
#     "/password-recovery-html-content/{email}",
#     dependencies=[Depends(has_authorization)],
#     response_class=HTMLResponse,
# )
# def recover_password_html_content(request: Request, email: str) -> Any:
#     """
#     HTML Content for Password Recovery
#     """
#     with request.app.state.postgres.get_session() as session:
#         user = request.app.state.postgres.get_user_by_email(session=session, email=email)

#         if not user:
#             raise HTTPException(
#                 status_code=404,
#                 detail="The user with this username does not exist in the system.",
#             )
#         password_reset_token = generate_password_reset_token(email=email)
#         email_data = generate_reset_password_email(
#             email_to=user.email, email=email, token=password_reset_token
#         )

#         return HTMLResponse(
#             content=email_data.html_content, headers={"subject:": email_data.subject}
#         )
