from __future__ import annotations
from fastapi import UploadFile

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import dataclasses
import os
import shutil
from minio_client import MinioConnection

class CharacterDownloadInputs():
    name: str
    knowledge_file_name: str
    avatar_file_name: str
    audio_file_name: str

class CharacterDownloadOutputs(BaseModel):
    avatar_file: bytes
    knowledge_file: bytes
    audio_file: bytes

class CharacterDownloadMinioService(BaseService):
    minio_client: MinioConnection

    async def process(self, character_download_inputs: CharacterDownloadInputs):
        try:
            self.minio_client.make_bucket("reunion")
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)

            # Download knowledge file
            temp_file_path = os.path.join(temp_dir, character_download_inputs.knowledge_file_name)
            knowledge_url = self.minio_client.get_object(bucket_name="reunion",
                                                                    folder_name=character_download_inputs.name,
                                                                    file_name=character_download_inputs.knowledge_file_name,
                                                                    local_file_name=temp_file_path)
            knowledge_file = None
            with open(temp_file_path, "wb") as buffer:
                knowledge_file = buffer.read()
            os.remove(temp_file_path)
            
            # Download avatar file
            temp_file_path = os.path.join(temp_dir, character_download_inputs.avatar_file_name)
            avatar_url = self.minio_client.get_object(bucket_name="reunion",
                                                                folder_name=character_download_inputs.name,
                                                                file_name=character_download_inputs.avatar_file_name,
                                                                local_file_name=temp_file_path)
            avatar_file = None
            with open(temp_file_path, "wb") as buffer:
                avatar_file = buffer.read()
            os.remove(temp_file_path)
            
            # Download audio file
            temp_file_path = os.path.join(temp_dir, character_download_inputs.audio_file_name)
            audio_url = self.minio_client.put_object(bucket_name="reunion",
                                                                folder_name=character_download_inputs.name,
                                                                file_name=character_download_inputs.audio_file_name,
                                                                local_file_name=temp_file_path)
            audio_file = None
            with open(temp_file_path, "wb") as buffer:
                audio_file = buffer.read()
            os.remove(temp_file_path)

            os.removedirs(temp_dir)
            
            return CharacterDownloadOutputs(
                avatar_file=avatar_file,
                knowledge_file=knowledge_file,
                audio_file=audio_file,
            )
        
        except Exception as e:
            raise e

