from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv
from dotenv import load_dotenv

from minio_client import MinioSettings
from mongo_client import MongoSettings
from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import YamlConfigSettingsSource
from pydantic import computed_field
from pydantic import model_validator
from pydantic.networks import EmailStr

load_dotenv(find_dotenv('.env'), override=True)

class Settings(BaseSettings):
    mongodb: MongoSettings
    minio: MinioSettings
    indexer_service_url: str
    
    FRONTEND_HOST: str
    PROJECT_NAME: str
    
    SMTP_TLS: bool 
    SMTP_SSL: bool 
    SMTP_PORT: int 
    SMTP_HOST: str | None
    SMTP_USER: str | None 
    SMTP_PASSWORD: str | None 
    EMAILS_FROM_EMAIL: EmailStr | None
    EMAILS_FROM_NAME: EmailStr | None = None
    
    @model_validator(mode="after")
    def _set_default_emails_from(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME
        return self

    @computed_field
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    class Config:
        env_nested_delimiter = '__'
        yaml_file = str(Path(__file__).parent.parent.parent / 'settings.yaml')
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls),
        )
