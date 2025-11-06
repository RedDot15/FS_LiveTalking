from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import dataclasses
import os
import shutil
from chromadb_client import ChromaDB
from indexer.shared.tools import generate_index_id
import datetime

class ChromaDBUploadInputs(BaseModel):
    character_name: str
    character_id: str
    embeddings: list
    chunks: list[str]

class CharacterUploadChromaDBService(BaseService):
    chromadb: ChromaDB
    
    async def process(self, inputs: ChromaDBUploadInputs):
        metadatas = [{"source": inputs.character_name} for _ in inputs.chunks]
        ids = [await generate_index_id(f"{inputs.character_name}{datetime.datetime.now()}{i}") for i  in range(len(inputs.chunks))]
        try:
            self.chromadb.add_document(
                character_id=inputs.character_id,
                documents=inputs.chunks,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            raise e
        
        
        return True
