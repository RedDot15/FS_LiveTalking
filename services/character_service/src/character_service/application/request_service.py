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
from logger import get_logger

from mongo_client.controller import CharacterHandler
from mongo_client.controller.request import RequestHandler
from mongo_client.model.entity import Request

from character_service.domain.upload_minio import CharacterUploadMinioService, CharacterInputs

logger = get_logger(__name__)

class RequestInput(BaseModel):
    character_name: str = Form(...)
    character_avatar_image: UploadFile = File(...)
    character_knowledge_file: UploadFile = File(...)
    character_audio_file: UploadFile = File(...)

class RequestOutput(BaseModel):
    request_id: str
    character_id: str
    character_name: str

class RequestRejectInput(BaseModel):
    request_id: str = None
    reject_reason: str = "Không đạt tiêu chuẩn"

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
    def delete_minio_init(self) -> CharacterDeleteMinioService:
        return CharacterDeleteMinioService(
            minio_client = self.request.app.state.minio_client
        )
        
    async def add_creation_request(self, inputs: RequestInput, current_user_id: str) -> RequestOutput:
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                character_id = str(uuid4())
                # Save image, knowledge, voice -> minio
                logger.info("========Minio process===========")
                character_outputs = await self.upload_minio_init.process(
                    character_inputs = CharacterInputs(
                        id = character_id,
                        knowledge_file = inputs.character_knowledge_file,
                        avatar_image = inputs.character_avatar_image,
                        audio_file=inputs.character_audio_file,
                    )
                )

                request_id = str(uuid4())

                request_handler = RequestHandler(collection=mongodb["requests"])
                request_handler.create_request(Request(
                    _id=request_id, 
                    character_id=character_id,
                    character_name=inputs.character_name, 
                    knowledge_url=character_outputs.knowledge_url,
                    avatar_url=character_outputs.avatar_url,
                    audio_url=character_outputs.audio_url,
                    created_at=datetime.now(), 
                    created_by=current_user_id, 
                    status="PENDING"))
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
        
        return RequestOutput(request_id=request_id, character_name=inputs.character_name, character_id=character_id)
    
    async def approve_request(self, request_id: str, current_user_id: str):
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                request_handler = RequestHandler(collection=mongodb["requests"])
                request = request_handler.get_request_by_id(request_id=request_id)
                if not request:
                    raise Exception(f"Request not found: {request_id}")
                if request['status'] != "PENDING":
                    raise Exception(f"Request status is not PENDING: {request_id}")

                # Request indexer
                await request_indexer(
                    self.request,
                    character_name = request['character_name'],
                    character_id=request['character_id'],
                    knowledge_url = request['knowledge_url'],
                    avatar_url = request['avatar_url'],
                    audio_url = request['audio_url'],
                )

                request['status'] = "APPROVED"
                request['approved_by'] = current_user_id
                request['evaluated_at'] = datetime.now()
                request_handler.update_request_by_id(request_id = request_id, request = request)
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
                if request['status'] != "PENDING":
                    raise Exception(f"Request status is not PENDING: {inputs.request_id}")

                # Delete image, knowledge, voice from minio
                await self.delete_minio_init.process(
                    character_delete_inputs = CharacterDeleteInputs(
                        id = request['character_id']
                    )
                )

                request['status'] = "REJECTED"
                request['rejected_by'] = current_user_id
                request['evaluated_at'] = datetime.now()
                request['reject_reason'] = inputs.reject_reason
                request_handler.update_request_by_id(request_id = inputs.request_id, request = request)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
        
        return

    def process(self, inputs: Any) -> Any:
        pass