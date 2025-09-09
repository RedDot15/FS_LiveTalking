from __future__ import annotations

from base import BaseModel, BaseService
from fastapi import FastAPI, Request, File, UploadFile, Depends

from typing import Annotated, Any
from pydantic import ConfigDict, Field

from indexer.domain.parser import ParserService
from indexer.domain.upload_minio import CharacterUploadMinioService, CharacterInputs, CharacterOutputs
from indexer.domain.gen_uuid import GenUUIDService, CharacterIdInputs, CharacterIdOutputs
from indexer.domain.upload_mongodb import CharacterUploadMongoDBService, CharacterMongoDBInputs, CharacterMongoDBOutputs
from indexer.domain.parser import ParserService, ParserInput, ParserOutput


class IndexerApplication(BaseService):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]

    @property
    def upload_minio_init(self) -> CharacterUploadMinioService:
        return CharacterUploadMinioService(
            minio_client = self.request.app.state.minio_client
        )
    
    @property
    def gen_uuid_init(self) -> GenUUIDService:
        return GenUUIDService()
    
    @property
    def upload_mongodb_init(self) -> CharacterUploadMongoDBService:
        return CharacterUploadMongoDBService(
            mongodb_handler = self.request.app.state.mongodb_handler
        )
    
    @property
    def parse_file_init(self) -> ParserService:
        return ParserService()

    # nghich luon
    async def process(self, inputs: CharacterInputs) -> CharacterOutputs:
        minio_response = await self.upload_minio_init.process(
            character_inputs = CharacterInputs(
                name = inputs.name,
                knowledge_file = inputs.knowledge_file,
                avatar_image = inputs.avatar_image,
            )
        )
        return CharacterOutputs(
            avatar_url = minio_response.avatar_url,
            knowledge_url = minio_response.knowledge_url,
        )
    async def gen_uuid(self, inputs: CharacterIdInputs) -> CharacterIdOutputs:
        character_id = await self.gen_uuid_init.process(
            character_id=CharacterIdInputs(
                character_id=inputs.name
            )
        )
        return CharacterIdOutputs(character_id=character_id)
    
    async def upload_to_mongodb(self, inputs: CharacterMongoDBInputs) -> CharacterMongoDBOutputs:
        result = await self.upload_mongodb_init.upload_to_mongo(
            inputs=CharacterMongoDBInputs(
                name=inputs.name,
                avatar_url=inputs.avatar_url,
            )
        )
        return CharacterMongoDBOutputs(result=result.inserted_id)
    
    async def parse_file(self, inputs: ParserInput) -> ParserOutput:
        md_text = await self.parse_file_init.process(
            inputs=inputs
        )
        return ParserOutput(parsed_text=md_text)        