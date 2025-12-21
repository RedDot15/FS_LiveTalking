from __future__ import annotations

import json

import httpx
from logger import get_logger
from fastapi import Request


logger = get_logger(__name__)

async def request_delete_datas(user_id: str, request: Request) -> list[str]:
    
    authorization_header = request.headers.get('Authorization')
    
    headers = {}
    if authorization_header:
        headers['Authorization'] = authorization_header

    try:
        async with httpx.AsyncClient(timeout=6000) as client:
            response = await client.delete(
                url=f"http://character_service:3006/v1/users/{user_id}/mongo_datas",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(
                    f'API request failed with status {response.status_code}: {response.text}',
                )
                return []
            
            return
        
    except httpx.RequestError as e:
        logger.exception(f'Network error while fetching context: {e}')
        return []
    except json.JSONDecodeError as e:
        logger.exception(f'Failed to decode JSON response: {e}')
        return []
    except Exception as e:
        logger.exception(f'Unexpected error in get_context: {e}')
        return []