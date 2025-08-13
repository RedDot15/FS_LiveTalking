from __future__ import annotations

import json
import httpx

from functools import lru_cache

from logger import get_logger

from ..utils import get_settings

logger = get_logger(__name__)

async def get_context(question: str) -> list[str]:
    if not question.strip():
        logger.warning('Empty question provided to get_context')
        return []
    
    try:
        settings = get_settings()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=settings.rag_service_url,
                json={'query': question},
            )
            
            if response.status_code != 200:
                logger.warning(
                    f'API request failed with status {response.status_code}: {response.text} : {settings.rag_service_url}',
                )
                return []
            
            response_data = response.json()
            
            search_output = response_data.get('info', {}).get('results', [])
            
            return search_output
        
    except httpx.RequestError as e:
        logger.exception(f'Network error while fetching context: {e}')
        return []
    except json.JSONDecodeError as e:
        logger.exception(f'Failed to decode JSON response: {e}')
        return []
    except Exception as e:
        logger.exception(f'Unexpected error in get_context: {e}')
        return []