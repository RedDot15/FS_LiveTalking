from __future__ import annotations

from pydantic import BaseModel
from chromadb_client import ChromaDB

class RagServiceInput(BaseModel):
    query: str 
    
    
class RagServiceOutput(BaseModel):
    results: list[str]
    
    
class RagServiceApplication:
    
    chromadb: ChromaDB
    
    def process(self, input: RagServiceInput) -> RagServiceOutput:
        pass