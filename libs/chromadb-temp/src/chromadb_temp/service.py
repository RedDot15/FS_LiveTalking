from __future__ import annotations

from .settings import ChromaDBSetting

import chromadb
from base import BaseModel
from base import BaseService
from chromadb.config import Settings

class ChromaDBInput(BaseModel):
    
    query: list
    topk: int = 5
    document_collections: str
    
class ChromaDBOutput(BaseModel):
    results: list[str]


class ChromaDB(BaseService):
    
    chromadb_setting: ChromaDBSetting
    
    @property
    def client(self) -> chromadb.HttpClient:
        return chromadb.HttpClient(
            host=self.chromadb_setting.host,
            port=self.chromadb_setting.port,
            settings=Settings(
                allow_reset=self.chromadb_setting.allow_reset,
                anonymized_telemetry=self.chromadb_setting.anonymized_telemetry
            )
        )
        
    def collections(self, documen_collections: str):
        return self.client.get_or_create_collection(
            name=documen_collections
        )
        
    def store_vector(
            self, 
            document_collections: str, 
            vectors: list[list[float]], 
            metadatas: list[dict], 
            ids: list,
            documents: list[str],
        ):

        
        self.collections(documen_collections=document_collections).upsert(
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def process(self, inputs: ChromaDBInput) -> ChromaDBOutput:
        results = self.collections(documen_collections=inputs.document_collections).query(
            query_embeddings=inputs.query,
            n_results=inputs.topk,
        )

        documents = results.get("documents", [[]])[0]
        documents = [doc for doc in documents if doc is not None]

        return ChromaDBOutput(results=documents)