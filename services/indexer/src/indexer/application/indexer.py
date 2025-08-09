from __future__ import annotations

from pydantic import BaseModel
from fastapi import Request

from indexer.domain.parser import ParserService

class IndexerApplication(BaseModel):
    request: Request
    file_extension: str = 'pdf'
    
    @property
    def parser(self) -> ParserService:
        return ParserService()