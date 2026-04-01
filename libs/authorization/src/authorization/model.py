
from sqlmodel import SQLModel

# Contents of JWT token
class TokenPayload(SQLModel):
    id: str | None = None
    scope: str | None = None
    username: str | None = None
    email: str | None = None