from __future__ import annotations

from .settings import ChromaDBSetting

import chromadb
from base import CustomBaseModel
from base import BaseService
from chromadb.config import Settings
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class ChromaDBInput(CustomBaseModel):
    query: str
    topk: int = 10
    
class ChromaDBOutput(CustomBaseModel):
    results: list[str]


class ChromaDB(BaseService):
    chromadb_setting: ChromaDBSetting
    
    @property
    def client(self) -> chromadb.HttpClient:
        http_client = chromadb.HttpClient(
            host=self.chromadb_setting.host,
            port=self.chromadb_setting.port,
            settings=Settings(
                allow_reset=self.chromadb_setting.allow_reset,
                anonymized_telemetry=self.chromadb_setting.anonymized_telemetry
            )
        )
        
        return http_client.get_or_create_collection(
            name=self.chromadb_setting.document_collections,
            # embedding_function=SentenceTransformerEmbeddingFunction(
            #     model_name=self.chromadb_setting.model_name
            # )
        )
    
    def add_document(self, documents: list[str], metadatas: list[dict], ids: list):
        self.client.add(
            documents=documents, 
            metadatas=metadatas, 
            ids=ids
        )
        
    def process(self, input: ChromaDBInput) -> ChromaDBOutput:
        results = self.client.query(
            query_texts=input.query,
            n_results=input.topk
        )

        return ChromaDBOutput(results=results['documents'][0])
