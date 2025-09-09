from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
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
    ):
        self.name = name
        self.knowledge_file = knowledge_file
        self.avatar_image = avatar_image
# class CharacterInputs(BaseModel):
#     name: str = Form(...)
#     knowledge_file: UploadFile = File(...)
#     avatar_image: UploadFile = File(...)

class CharacterOutputs(BaseModel):
    avatar_url: str
    knowledge_url: str

class CharacterUploadMinioService(BaseService):
    minio_client: MinioConnection

    async def process(self, character_inputs: CharacterInputs):
        try:
            self.minio_client.make_bucket("character")
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
            knowledge_url = self.minio_client.put_object_from_local_path(bucket_name="character",
                                                                    src_file=temp_file_path,
                                                                    des_folder_name=object_name,
                                                                    des_file=file_name)
            os.remove(temp_file_path)
            
            avatar: UploadFile = character_inputs.avatar_image
            temp_file_path = os.path.join(temp_dir, avatar.filename)
            # Lưu file người dùng upload vào thư mục tạm
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(avatar.file, buffer)

            avatar_url = self.minio_client.put_object_from_local_path(bucket_name="character",
                                                                src_file=temp_file_path,
                                                                des_folder_name=object_name,
                                                                des_file=avatar.filename)

            # Xóa file tạm sau khi upload
            os.remove(temp_file_path)
            return CharacterOutputs(
                avatar_url=avatar_url,
                knowledge_url=knowledge_url
            )
        
        except Exception as e:
            raise e
