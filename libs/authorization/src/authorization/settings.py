
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str
    SECRET_KEY: str 
    ALGORITHM: str 


settings = Settings()