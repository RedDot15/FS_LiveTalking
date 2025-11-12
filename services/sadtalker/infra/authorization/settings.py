
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    ACCESS_TOKEN_SECRET_KEY: str 
    ALGORITHM: str 


settings = Settings()