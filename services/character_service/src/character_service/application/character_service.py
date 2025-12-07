from __future__ import annotations
from uuid import uuid4

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from mongo_client.controller import CharacterHandler

from character_service.domain.upload_minio import CharacterUploadMinioService

class CharacterServiceOutput(BaseModel):
    characters: list[dict]

class CharacterServiceApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]

    @property
    def upload_minio_init(self) -> CharacterUploadMinioService:
        return CharacterUploadMinioService(
            minio_client = self.request.app.state.minio_client
        )

    def process(self) -> CharacterServiceOutput:
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                char_handler = CharacterHandler(collection=mongodb["characters"])
                characters = char_handler.get_character()
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
            
        return CharacterServiceOutput(characters=characters)