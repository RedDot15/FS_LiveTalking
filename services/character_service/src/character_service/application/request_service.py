from __future__ import annotations
from datetime import datetime
from uuid import uuid4

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from character_service.domain.delete_minio.service import CharacterDeleteInputs, CharacterDeleteMinioService
from character_service.shared.tools import request_indexer
from fastapi import File, Form, UploadFile
from pydantic import ConfigDict
from pydantic import Field

from mongo_client.controller import CharacterHandler
from mongo_client.controller.request import RequestHandler
from mongo_client.model.entity import Request

from character_service.domain.upload_minio import CharacterUploadMinioService, CharacterInputs
from character_service.domain.download_minio.service import CharacterDownloadInputs, CharacterDownloadMinioService

class RequestInput(BaseModel):
    character_name: str = Form(...),
    character_avatar_image: UploadFile = File(...),
    character_knowledge_file: UploadFile = File(...),
    character_audio_file: UploadFile = File(...),

class RequestOutput(BaseModel):
    character_name: str

class RequestRejectInput(BaseModel):
    request_id: str
    reject_reason: str

class RequestServiceApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]

    @property
    def upload_minio_init(self) -> CharacterUploadMinioService:
        return CharacterUploadMinioService(
            minio_client = self.request.app.state.minio_client
        )
    
    @property
    def download_minio_init(self) -> CharacterDownloadMinioService:
        return CharacterDownloadMinioService(
            minio_client = self.request.app.state.minio_client
        )
    
    @property
    def delete_minio_init(self) -> CharacterDeleteMinioService:
        return CharacterDeleteMinioService(
            minio_client = self.request.app.state.minio_client
        )
        
    async def add_creation_request(self, inputs: RequestInput, current_user_id: str) -> RequestOutput:
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                character_handler = CharacterHandler(collection=mongodb["characters"])
                character = character_handler.get_character_by_name(character_name=inputs.character_name)
                if character:
                    raise Exception(f"Character already exists {inputs.character_name}")

                request_handler = RequestHandler(collection=mongodb["requests"])
                request = request_handler.get_request_by_character_name(character_name=inputs.character_name)
                if request:
                    raise Exception(f"Request already exists {inputs.character_name}")

                # Save image, knowledge, voice -> minio
                character_outputs = await self.upload_minio_init.process(
                    character_inputs = CharacterInputs(
                        name = inputs.character_name,
                        knowledge_file = inputs.character_knowledge_file,
                        avatar_image = inputs.character_avatar_image,
                        audio_file=inputs.character_audio_file,
                    )
                )

                request_handler.create_request(Request(
                    _id=uuid4(), 
                    character_name=inputs.character_name, 
                    created_at=datetime.now(), 
                    created_by=current_user_id, 
                    knowledge_file_info=character_outputs.knowledge_file_info,
                    image_file_info=character_outputs.avatar_file_info,
                    audio_file_info=character_outputs.audio_file_info,
                    status="PENDING"))
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
        
        return RequestOutput(character_name=inputs.character_name)
    
    async def approve_request(self, request_id: str, current_user_id: str):
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                request_handler = RequestHandler(collection=mongodb["requests"])
                request = request_handler.get_request_by_id(request_id=request_id)
                if not request:
                    raise Exception(f"Request not found: {request_id}")

                # Download image, knowledge, voice from minio
                character_download_outputs = await self.download_minio_init.process(
                    character_download_inputs = CharacterDownloadInputs(
                        name = request.character_name,
                        knowledge_file_name = request.knowledge_file_info.filename,
                        avatar_file_name = request.image_file_info.filename,
                        audio_file_name=request.audio_file_info.filename,
                    )
                )

                # Request indexer
                request_indexer(
                    name = request.character_name,
                    knowledge_file = character_download_outputs.knowledge_file,
                    knowledge_file_info = request.knowledge_file_info,
                    avatar_file = character_download_outputs.avatar_file,
                    avatar_file_info = request.image_file_info,
                    audio_file = character_download_outputs.audio_file,
                    audio_file_info = request.audio_file_info
                )

                request_handler = RequestHandler(collection=mongodb["requests"])
                request_handler.update_request_by_id(request_id = request_id, request = Request(
                    status="APPROVED", 
                    approved_by=current_user_id,
                    approved_at=datetime.now()))
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
        
        return
        
    async def reject_request(self, inputs: RequestRejectInput, current_user_id: str):
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                request_handler = RequestHandler(collection=mongodb["requests"])
                request = request_handler.get_request_by_id(request_id=inputs.request_id)
                if not request:
                    raise Exception(f"Request not found: {inputs.request_id}")

                # Delete image, knowledge, voice from minio
                await self.delete_minio_init.process(
                    character_delete_inputs = CharacterDeleteInputs(
                        name = request.character_name
                    )
                )

                request_handler = RequestHandler(collection=mongodb["requests"])
                request_handler.update_request_by_id(request_id = inputs.request_id, request = Request(
                    status="REJECTED", 
                    rejected_by=current_user_id,
                    reject_reason=inputs.reject_reason))
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
        
        return
        