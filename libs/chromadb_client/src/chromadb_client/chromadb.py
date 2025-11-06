from __future__ import annotations

from .settings import ChromaDBSetting

import chromadb
from base import BaseModel
from base import BaseService
from chromadb.config import Settings
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class ChromaDBInput(BaseModel):
    character_id: str
    query: str
    topk: int = 10
    
class ChromaDBOutput(BaseModel):
    results: list[str]


class ChromaDB(BaseService):
    chromadb_setting: ChromaDBSetting
    
    @property
    def client(self) -> chromadb.ClientAPI:
        return chromadb.HttpClient(
            host=self.chromadb_setting.host,
            port=self.chromadb_setting.port,
            settings=Settings(
                allow_reset=self.chromadb_setting.allow_reset,
                anonymized_telemetry=self.chromadb_setting.anonymized_telemetry
            )
        )
    
    def add_document(self, character_id: str, documents: list[str], metadatas: list[dict], ids: list):
        self.client.get_or_create_collection(name=character_id).add(
                documents=documents, 
                metadatas=metadatas, 
                ids=ids
        )
        
    def process(self, input: ChromaDBInput) -> ChromaDBOutput:
        results = self.client.get_or_create_collection(name=input.character_id).query(
            query_texts=input.query,
            n_results=input.topk
        )

        return ChromaDBOutput(results=results['documents'][0])