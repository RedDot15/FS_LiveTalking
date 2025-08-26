from pydantic import BaseModel, SecretStr

class MinioSettings(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
