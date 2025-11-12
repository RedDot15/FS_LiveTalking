from __future__ import annotations

import json

import httpx
from logger import get_logger

from ..utils import get_settings

logger = get_logger(__name__)

async def request_livetalking_echo(message: str, sessionid: int):
    if not message.strip():
        logger.warning('Empty question provided to get_context')
        return []
    
    try:
        settings = get_settings()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=settings.livetalking_service_url + '/human',
                json={
                    'sessionid': sessionid,
                    'type': 'echo',
                    'text': message
                },
            )
            
            if response.status_code != 200:
                logger.warning(
                    f'API request failed with status {response.status_code}: {response.text} : {settings.livetalking_service_url}',
                )
                return []
            
    except httpx.RequestError as e:
        logger.exception(f'Network error while fetching context: {e}')
        return []
    except json.JSONDecodeError as e:
        logger.exception(f'Failed to decode JSON response: {e}')
        return []
    except Exception as e:
        logger.exception(f'Unexpected error in get_context: {e}')
        return []
    
