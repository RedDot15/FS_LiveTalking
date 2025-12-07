from __future__ import annotations

from base import BaseModel, BaseService
from logger import get_logger
from minio_client import MinioConnection

class CharacterDeleteInputs(BaseModel):
    id: str

class CharacterDeleteMinioService(BaseService):
    minio_client: MinioConnection

    async def process(self, character_delete_inputs: CharacterDeleteInputs):
        try:
            self.minio_client.make_bucket("reunion")
            self.minio_client.remove_folder(bucket_name="reunion",
                                            folder_name=character_delete_inputs.id)
            return
        
        except Exception as e:
            raise e

