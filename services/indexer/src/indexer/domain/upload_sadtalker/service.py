from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
import httpx, asyncio
import json
from indexer.shared.utils import get_settings
from fastapi import Request
from logger import get_logger

logger = get_logger(__name__)

# ===== Tam thoi hardcode Sadtalker ========
class SadTalkerServiceInput(BaseModel):
    character_id: str 
    image_url: str
    audio_url: str

class SadTalkerServiceOutput(BaseModel):
    save_path: str


class UploadSadtalkerService(BaseService):

    async def process(self, inputs: SadTalkerServiceInput) -> str:
        url = get_settings().sadtalker_service_url 
        
        data = {
            'character_id': inputs.character_id,
            'image_url': inputs.image_url,
            'audio_url': inputs.audio_url
        }

        try:
            async with httpx.AsyncClient(timeout=6000) as client:
                response = await client.post(url, json=data)

                response.raise_for_status()    
                response_data = response.json() 
                save_path = response_data.get('info').get('save_path')
                return save_path
        
        except Exception as e:
            raise e
        