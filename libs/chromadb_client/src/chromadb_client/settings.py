from __future__ import annotations

from chromadb.config import Settings

class ChromaDBSetting:
    host: str 
    port: str 
    document_collections: str = 'reunion'
    allow_reset: bool = True
    anonymized_telemetry: bool = True
    

    