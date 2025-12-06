from __future__ import annotations

import json

import httpx
from logger import get_logger
from fastapi import Request

from ..utils import get_settings

logger = get_logger(__name__)

async def request_indexer(
        request: Request,
        character_name: str,
        character_id: str, 
        knowledge_url: str,
        avatar_url: str, 
        audio_url: str):
    
    try:
        settings = get_settings()

        
        authorization_header = request.headers.get('Authorization')
        
        headers = {}
        if authorization_header:
            headers['Authorization'] = authorization_header

        # 1. Define the form fields (non-file data)
        data = {
            'id': character_id,
            'name': character_name,
            'knowledge_url': knowledge_url,
            'avatar_url': avatar_url,
            'audio_url': audio_url
        }
        

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=settings.indexer_service_url + '/indexing',
                json=data,
                headers=headers
            )
            
            if response.status_code != 200:
                raise Exception(f'API request failed with status {response.status_code}: {response.text}')
            
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
    
