from __future__ import annotations

from base import BaseModel
from base import BaseService
from chromadb_client import ChromaDB
from chromadb_client import ChromaDBInput

from fastapi import Request


class RagServiceInput(BaseModel):
    query: str 
    
    
class RagServiceOutput(BaseModel):
    results: list[str] | None
    
    
class RagServiceApplication(BaseService):
    
    request: Request
    
    @property
    def chromadb(self) -> ChromaDB:
        return ChromaDB(chromadb_setting=self.request.app.state.settings.chromadb)
    
    async def process(self, input: RagServiceInput) -> RagServiceOutput:
        
        results = self.chromadb.process(
            input=ChromaDBInput(
                query=input.query,
                topk=3
            )
        )
        
        return RagServiceOutput(results=results.results)