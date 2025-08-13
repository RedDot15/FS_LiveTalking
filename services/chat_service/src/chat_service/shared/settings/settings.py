from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv
from dotenv import load_dotenv

from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import YamlConfigSettingsSource

from llm import LLMSetting
from .answer_aggregator import AnswerAggregatorSettings

load_dotenv(find_dotenv('.env'), override=True)

class Settings(BaseSettings):

    rag_service_url: str
    llm: LLMSetting
    answer_aggregator_settings: AnswerAggregatorSettings
    
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
