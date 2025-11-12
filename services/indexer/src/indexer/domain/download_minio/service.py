from __future__ import annotations
from fastapi import UploadFile

import requests
from base import BaseModel, BaseService
from logger import get_logger
from pydantic import dataclasses
import os
import shutil
from minio_client import MinioConnection
import tempfile

class CharacterDownloadInputs():
    character_id: str
    knowledge_url: str

class CharacterDownloadOutputs(BaseModel):
    knowledge_file: bytes

class CharacterDownloadMinioService(BaseService):
    minio_client: MinioConnection

    async def process(self, character_download_inputs: CharacterDownloadInputs):
        try:
            return get_knowledge_file(
                bucket_name = "reunion", 
                character_id = character_download_inputs.character_id, 
                knowledge_url = character_download_inputs.knowledge_url)
        except Exception as e:
            raise e

    def get_knowledge_file(self, bucket_name: str, character_id: str, knowledge_url: str):
        
        logger.info(
            'STARTING TO GENERATE PRESIGNED URLS FROM MINIO',
            extra={'bucket_name': bucket_name, 'knowledge_url': knowledge_url}
        )
        
        knowledge_url_minio = character_id + '/' + knowledge_url

        presigned_knowledge_url = self.minio_client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=knowledge_url_minio 
        )

        logger.info('DONE GENERATING PRESIGNED URLS')

        def download_to_temp(url, original_filename):
            
            suffix = os.path.splitext(original_filename)[1] or ''
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_file.flush()
                temp_file_path = temp_file.name
                temp_file.close()
                return temp_file_path
            finally:
                temp_file.close()

        knowledge_file_local_path = download_to_temp(presigned_knowledge_url, knowledge_url_minio)

        return knowledge_file_local_path