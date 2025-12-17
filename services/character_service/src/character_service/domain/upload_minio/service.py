from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import dataclasses
import os
import shutil
from minio_client import MinioConnection, FilePathStatus

logger = get_logger(__name__)

class CharacterInputs():
    def __init__(
        self,
        id: str = Form(...),
        knowledge_file: UploadFile = File(...),
        avatar_image: UploadFile = File(...),
        audio_file: UploadFile = File(...),
    ):
        self.id = id
        self.knowledge_file = knowledge_file
        self.avatar_image = avatar_image
        self.audio_file = audio_file

class CharacterOutputs(BaseModel):
    avatar_url: str
    knowledge_url: str
    audio_url: str

class CharacterUploadMinioService(BaseService):
    minio_client: MinioConnection

    def get_final_url(self, full_path: str) -> str:
        parts = full_path.split('/')
        last_two_parts = parts[-2:]
        return "/".join(last_two_parts)

    async def process(self, character_inputs: CharacterInputs):
        try:
            self.minio_client.make_bucket("reunion")
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            character_id = character_inputs.id
            # ================================KNOWLEDGE================================
            # Đường dẫn file tạm
            file: UploadFile = character_inputs.knowledge_file
            file_name = file.filename
            temp_file_path = os.path.join(temp_dir, file_name)

            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Upload file knowledge lên MinIO 
            file_type = "knowledges"
            file_name = f"{file_type}/{file_name}"
            knowledge_status = self.minio_client.put_object(bucket_name="reunion",
                                                        src_file=temp_file_path,
                                                        des_folder_name=character_id,
                                                        des_file_name=file_name)
            if knowledge_status.status:
                knowledge_url = knowledge_status.full_path
            await file.seek(0)
            os.remove(temp_file_path)
            # ================================IMAGES================================
            avatar_image: UploadFile = character_inputs.avatar_image
            file_name = avatar_image.filename
            temp_file_path = os.path.join(temp_dir, avatar_image.filename)
            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(avatar_image.file, buffer)
            file_type = "images"
            file_name = f"{file_type}/{file_name}"

            avatar_status = self.minio_client.put_object(bucket_name="reunion",
                                                        src_file=temp_file_path,
                                                        des_folder_name=character_id,
                                                        des_file_name=file_name)

            if avatar_status.status:
                avatar_url=avatar_status.full_path
            await avatar_image.seek(0)
            os.remove(temp_file_path)
            # ================================AUDIOS================================
            audio_file: UploadFile = character_inputs.audio_file
            file_name = audio_file.filename
            temp_file_path = os.path.join(temp_dir, audio_file.filename)
            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)
            file_type = "audios"
            file_name = f"{file_type}/{file_name}"

            audio_status = self.minio_client.put_object(bucket_name="reunion",
                                                                src_file=temp_file_path,
                                                                des_folder_name=character_id,
                                                                des_file_name=file_name)
            if audio_status.status:
                audio_url=audio_status.full_path
            await audio_file.seek(0)
            os.remove(temp_file_path)
            os.removedirs(temp_dir)
            
            if avatar_status.status and knowledge_status.status and audio_status.status:
                return CharacterOutputs(
                    avatar_url=self.get_final_url(avatar_url),
                    knowledge_url=self.get_final_url(knowledge_url),
                    audio_url=self.get_final_url(audio_url),
                )
            else:
                raise Exception("Lỗi khi xử lí upload file lên Minio, File name đã tồn tại hoặc lỗi S3")
        
        except Exception as e:
            logger.error("Lỗi khi xử lí upload file lên Minio:", extra={'error': str(e)})
            raise e

