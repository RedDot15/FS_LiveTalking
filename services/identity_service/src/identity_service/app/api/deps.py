from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError

from identity_service.app.core import security
from identity_service.app.core.config import settings
from identity_service.app.models import TokenPayload

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)

TokenDep = Annotated[str, Depends(reusable_oauth2)]


def get_current_token(token: TokenDep) -> TokenPayload:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        return TokenPayload(**payload)
    except (InvalidTokenError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )


CurrentToken = Annotated[TokenPayload, Depends(get_current_token)]

def has_authority(authority: str):
    # This inner function is the actual dependency that FastAPI will execute.
    # It receives the token injected by 'get_current_token'.
    def has_permission(
        current_token: CurrentToken,
    ) -> TokenPayload:
        if authorization not in current_token.scope:
            raise HTTPException(
                status_code=403, detail="The user doesn't have enough privileges"
            )
        return current_token
    
    # Return the inner dependency function
    return has_permission
