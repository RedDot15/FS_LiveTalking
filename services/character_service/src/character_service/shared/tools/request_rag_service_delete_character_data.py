from __future__ import annotations

import json

import httpx
from logger import get_logger
from fastapi import Request

from ..utils import get_settings

logger = get_logger(__name__)

async def request_rag_service_delete_character_data(character_id: str, request: Request) -> str:
    
    authorization_header = request.headers.get('Authorization')
    
    headers = {}
    if authorization_header:
        headers['Authorization'] = authorization_header

    try:
        settings = get_settings()
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                url=settings.rag_service_url + character_id,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(
                    f'API request failed with status {response.status_code}: {response.text} : {settings.rag_service_url}',
                )
                raise Exception(f'API request failed with status {response.status_code}: {response.text}')
            
            return character_id
        
    except httpx.RequestError as e:
        raise Exception(f'Network error while fetching context: {e}')
    except json.JSONDecodeError as e:
        raise Exception(f'Failed to decode JSON response: {e}')
    except Exception as e:
        raise Exception(f'Unexpected error in get_context: {e}')