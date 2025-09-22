from __future__ import annotations

from base import CustomBaseModel
from base import BaseService
from chromadb_client import ChromaDB
from chromadb_client import ChromaDBInput

from fastapi import Request


class RagServiceInput(CustomBaseModel):
    topk: int
    query: str 
    
    
class RagServiceOutput(CustomBaseModel):
    results: list[str] | None
    
    
class RagServiceApplication(BaseService):
    
    request: Request

    async def process(self, input: RagServiceInput) -> RagServiceOutput:
        
        results = self.request.app.state.chromadb.process(
            input=ChromaDBInput(
                query=input.query,
                topk=input.topk
            )
        )
        
        return RagServiceOutput(results=results.results)