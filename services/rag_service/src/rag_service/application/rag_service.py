from __future__ import annotations

from base import BaseModel
from chromadb_client import ChromaDB
from chromadb_client import ChromaDBInput

from fastapi import Request


class RagServiceInput(BaseModel):
    query: str 
    
    
class RagServiceOutput(BaseModel):
    results: list[str]
    
    
class RagServiceApplication:
    
    request: Request
    
    @property
    def chromadb(self) -> ChromaDB:
        return ChromaDB(settings=self.request.app.state.settings.chromadb)
    
    def process(self, input: RagServiceInput) -> RagServiceOutput:
        
        results = self.chromadb.query(
            input=ChromaDBInput(
                query=input.query,
            )
        )
        
        return RagServiceOutput(results=results)