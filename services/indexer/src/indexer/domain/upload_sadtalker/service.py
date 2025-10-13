from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
import httpx, asyncio
import json
from indexer.shared.utils import get_settings

# ===== Tam thoi hardcode Sadtalker ========
class SadTalkerServiceInput(BaseModel):
    character_name: str 
    image_url: str
    audio_url: str

class SadTalkerServiceOutput(BaseModel):
    save_path: str


class UploadSadtalkerService(BaseService):

    async def process(self, inputs: SadTalkerServiceInput) -> str:
        url = get_settings().sadtalker_service_url 
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, json=inputs.model_dump())
                response.raise_for_status()    
                response_data = response.json() 
                save_path = response_data.get('info').get('save_path')
                return save_path
        
        except Exception as e:
            raise e
        