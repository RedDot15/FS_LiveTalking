from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
from pydantic import dataclasses
import os
import shutil
import httpx, asyncio
import json
from indexer.shared.utils import get_settings

class Wav2lipApplicationInput(BaseModel):
    character_name: str
    video_url: str
    
class Wav2lipApplicationOutput(BaseModel):
    wav2lip_result_path: str

class UploadWav2lipService(BaseService):

    async def process(self, inputs: Wav2lipApplicationInput) -> str:
        url = get_settings().wav2lip_service_url 
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, json=inputs.model_dump())
                response.raise_for_status()   
                response_data = response.json()
                wav2lip_result_path = response_data.get('wav2lip_result_data')  
                return wav2lip_result_path
        
        except Exception as e:
            raise e
        