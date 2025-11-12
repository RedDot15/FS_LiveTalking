from __future__ import annotations

import json

import httpx
from logger import get_logger

from ..utils import get_settings

logger = get_logger(__name__)

async def request_indexer(
        name: int, 
        knowledge_file: bytes,
        knowledge_file_info: dict,
        avatar_file: bytes, 
        avatar_file_info: dict,
        audio_file: bytes,
        audio_file_info: dict):
    
    try:
        settings = get_settings()

        # 1. Define the form fields (non-file data)
        data = {
            'name': name,
        }
        
        # 2. Define the files (key must match the FastAPI endpoint parameter name)
        files = {
            # ('filename', file_content, 'content_type')
            'knowledge_file': (knowledge_file_info.filename, knowledge_file, knowledge_file_info.content_type), 
            'avatar_image': (knowledge_file_info.filename, avatar_file, avatar_file_info.content_type),            
            'audio_file': (knowledge_file_info.filename, audio_file, audio_file_info.content_type),                  
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=settings.indexer_service_url + '/indexing',
                data=data,
                files=files,
            )
            
            if response.status_code != 200:
                logger.warning(
                    f'API request failed with status {response.status_code}: {response.text} : {settings.livetalking_service_url}',
                )
                return []
            
            return response.json()
            
    except httpx.RequestError as e:
        logger.exception(f'Network error while fetching context: {e}')
        return []
    except json.JSONDecodeError as e:
        logger.exception(f'Failed to decode JSON response: {e}')
        return []
    except Exception as e:
        logger.exception(f'Unexpected error in get_context: {e}')
        return []
    
