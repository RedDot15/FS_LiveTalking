from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import dataclasses
import os
import shutil
from minio_client import MinioConnection

class CharacterInputs():
    def __init__(
        self,
        name: str = Form(...),
        knowledge_file: UploadFile = File(...),
        avatar_image: UploadFile = File(...),
        audio_file: UploadFile = File(...),
    ):
        self.name = name
        self.knowledge_file = knowledge_file
        self.avatar_image = avatar_image
        self.audio_file = audio_file
# class CharacterInputs(BaseModel):
#     name: str = Form(...)
#     knowledge_file: UploadFile = File(...)
#     avatar_image: UploadFile = File(...)

class CharacterOutputs(BaseModel):
    avatar_url: str
    knowledge_url: str
    audio_url: str

class CharacterUploadMinioService(BaseService):
    minio_client: MinioConnection

    async def process(self, character_inputs: CharacterInputs):
        try:
            self.minio_client.make_bucket("reunion")
            file: UploadFile = character_inputs.knowledge_file
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)

            # Đường dẫn file tạm
            file_name = file.filename
            temp_file_path = os.path.join(temp_dir, file_name)

            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Upload file knowledge lên MinIO 
            object_name = character_inputs.name
            knowledge_url = self.minio_client.put_object(bucket_name="reunion",
                                                                    src_file=temp_file_path,
                                                                    des_folder_name=object_name,
                                                                    des_file_name=file_name)
            os.remove(temp_file_path)
            
            avatar_image: UploadFile = character_inputs.avatar_image
            temp_file_path = os.path.join(temp_dir, avatar_image.filename)
            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(avatar_image.file, buffer)

            avatar_url = self.minio_client.put_object(bucket_name="reunion",
                                                                src_file=temp_file_path,
                                                                des_folder_name=object_name,
                                                                des_file_name=avatar_image.filename)

            os.remove(temp_file_path)
            
            audio_file: UploadFile = character_inputs.audio_file
            temp_file_path = os.path.join(temp_dir, audio_file.filename)
            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            audio_url = self.minio_client.put_object(bucket_name="reunion",
                                                                src_file=temp_file_path,
                                                                des_folder_name=object_name,
                                                                des_file_name=audio_file.filename)

            os.remove(temp_file_path)
            os.removedirs(temp_dir)
            
            return CharacterOutputs(
                avatar_url=avatar_url,
                knowledge_url=knowledge_url,
                audio_url=audio_url,
            )
        
        except Exception as e:
            return CharacterOutputs(
                avatar_url="FAIL",
                knowledge_url="FAIL",
                audio_url="FAIL",
            )
            raise e

