from __future__ import annotations

from minio_client import MinioSettings
from mongo_client import MongoSettings
from chromadb_client import ChromaDBSetting
from litellm import LiteLLMSetting

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import YamlConfigSettingsSource

from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'), override=True)

class Settings(BaseSettings):
    minio: MinioSettings
    mongo: MongoSettings
    chromadb: ChromaDBSetting
    litellm: LiteLLMSetting
    sadtalker_service_url: str
    wav2lip_service_url: str

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
