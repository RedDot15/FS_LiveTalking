from __future__ import annotations

from base import BaseModel
from base import BaseService
from logger import get_logger
from chromadb_client import ChromaDB
from chromadb_client import ChromaDBInput

from fastapi import Request

logger = get_logger(__name__)

class RagServiceInput(BaseModel):
    character_id: str
    topk: int
    query: str 
    
class RagServiceOutput(BaseModel):
    results: list[str] | None
    
    
class RagServiceApplication(BaseService):
    
    request: Request

    async def process(self, input: RagServiceInput) -> RagServiceOutput:
        
        results = self.request.app.state.chromadb.process(
            input=ChromaDBInput(
                character_id=input.character_id,
                query=input.query,
                topk=input.topk
            )
        )
        
        logger.info(f'rag results: {results}')

        return RagServiceOutput(results=results.results)

    # async def delete_character_data(self, character_id: str) -> str:
        
    #     self.request.app.state.chromadb.delete_document(character_id=character_id)
        
    #     logger.info(f'Deleted character data of character: {character_id}')

    #     return character_id