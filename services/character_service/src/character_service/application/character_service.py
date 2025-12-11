from __future__ import annotations

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from mongo_client.controller import CharacterHandler

from character_service.domain.delete_minio.service import CharacterDeleteInputs, CharacterDeleteMinioService

from character_service.domain.upload_minio import CharacterUploadMinioService

from character_service.shared.tools import request_rag_service_delete_character_data

class CharacterServiceOutput(BaseModel):
    characters: list[dict]

class DeleteCharacterInput(BaseModel):
    character_id: str
    user_id: str

class DeleteCharacterOutput(BaseModel):
    character_id: str

class CharacterServiceApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]

    @property
    def upload_minio_init(self) -> CharacterUploadMinioService:
        return CharacterUploadMinioService(
            minio_client = self.request.app.state.minio_client
        )
        
    @property
    def delete_minio_init(self) -> CharacterDeleteMinioService:
        return CharacterDeleteMinioService(
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

    async def delete_character(self, character_id: str, user_id: str, scope: str) -> DeleteCharacterOutput:
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                char_handler = CharacterHandler(collection=mongodb["characters"])
                character = char_handler.get_character_by_id(character_id=character_id)

                if character is None:
                    raise Exception("Character not found")
                if "DELETE_CHARACTER" not in scope and character["created_by"] != user_id:
                    raise Exception("You are not the owner of this character")

                char_handler.delete_character_by_id(character_id=character_id)
                
                # Delete character's data from minio
                await self.delete_minio_init.process(
                    character_delete_inputs = CharacterDeleteInputs(
                        id = character_id
                    )
                )

                # Delete character's data from rag service
                await request_rag_service_delete_character_data(character_id=character_id)


            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
            
        return DeleteCharacterOutput(character_id=character_id)