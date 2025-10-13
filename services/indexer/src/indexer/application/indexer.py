from __future__ import annotations

from base import BaseModel, BaseService

from typing import Annotated, Any
from pydantic import ConfigDict, Field
import httpx, asyncio
from logger import get_logger

from indexer.domain.parser import ParserService
from indexer.domain.upload_minio import CharacterUploadMinioService, CharacterInputs
from indexer.domain.gen_uuid import GenUUIDService, CharacterIdInputs, CharacterIdOutputs
from indexer.domain.upload_mongodb import CharacterUploadMongoDBService, CharacterMongoDBInputs
from indexer.domain.parser import ParserService, ParserInput
from indexer.domain.upload_chromadb import CharacterUploadChromaDBService, ChromaDBUploadInputs
from indexer.domain.upload_sadtalker import SadTalkerServiceInput, UploadSadtalkerService
from indexer.domain.upload_wav2lip import UploadWav2lipService, Wav2lipApplicationInput

class IndexerApplicationOutput(BaseModel):
    json_response: str

logger = get_logger(__name__)

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
            db_handler = self.request.app.state.mongodb_handler
        )
    
    @property
    def parse_file_init(self) -> ParserService:
        return ParserService(
            litellm = self.request.app.state.litellm_service
        )
    
    @property
    def upload_chromadb_init(self)-> CharacterUploadChromaDBService:
        return CharacterUploadChromaDBService(
            chromadb = self.request.app.state.chroma_client
        )

    async def gen_uuid(self, inputs: CharacterIdInputs) -> CharacterIdOutputs:
        character_id = await self.gen_uuid_init.process(
            character_id=CharacterIdInputs(
                character_id=inputs.name
            )
        )
        return CharacterIdOutputs(character_id=character_id)
    
    async def upload_to_mongodb(self, inputs: CharacterMongoDBInputs):
        result = await self.upload_mongodb_init.process(
            inputs=CharacterMongoDBInputs(
                name=inputs.name,
                character_id=inputs.character_id,
            )
        )
        return result.result
    
    async def parse_file(self, inputs: ParserInput):
        embeddings = await self.parse_file_init.process(
            inputs=inputs
        )
        return embeddings

    async def process(self, inputs: CharacterInputs):
        logger.info("========Minio process===========")
        minio_response = await self.upload_minio_init.process(
            character_inputs = CharacterInputs(
                name = inputs.name,
                knowledge_file = inputs.knowledge_file,
                avatar_image = inputs.avatar_image,
                audio_file=inputs.audio_file,
            )
        )
        logger.info(f"Minio response:{minio_response}")
        character_genID = await self.gen_uuid_init.process(
            inputs=CharacterIdInputs(
                name=inputs.name
            )
        )
        logger.info(f"Character ID:{character_genID.character_id}")
        logger.info("========Mongodb process============")
        mongo_result = await self.upload_to_mongodb(
            inputs = CharacterMongoDBInputs(
                name = inputs.name,
                character_id=character_genID.character_id
            )
        )
        logger.info(f"Upload to Mongodb:{mongo_result}")
        logger.info("=========Embbeding process============")
        parse_output = await self.parse_file(
            inputs=ParserInput(
                knowledge_file=inputs.knowledge_file
            )
        )
        
        logger.info("======Chromadb process===========")
        upload_chromadb_status = await self.upload_chromadb_init.process(
            inputs=ChromaDBUploadInputs(
                character_name = inputs.name,
                character_id = character_genID.character_id,
                embeddings = parse_output.embeddings,
                chunks = parse_output.chunks
            )
        )
        logger.info(f"Upload Chromadb status: {upload_chromadb_status}")
        logger.info("========Upload Sadtalker==============")
        video_url = await UploadSadtalkerService().process(
            inputs = SadTalkerServiceInput(
                character_name = inputs.name,
                image_url = minio_response.avatar_url,
                audio_url = minio_response.audio_url,
            )
        )
        logger.info(f"Sadtalker video url: {video_url}")
        logger.info("========Upload Wav2lip==============")

        wav2lip_output = await UploadWav2lipService().process(
            inputs = Wav2lipApplicationInput(
                character_name = inputs.name,
                video_url = video_url,
            )
        )
        logger.info(f"Wav2lip video url: {wav2lip_output}")

        return IndexerApplicationOutput(
            json_response="Succeed"
        )
           