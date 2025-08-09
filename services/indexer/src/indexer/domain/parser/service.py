from __future__ import annotations

from pydantic import BaseModel
from fastapi import UploadFile

class ParserInput(BaseModel):
    file: UploadFile
    
class ParserOutput(BaseModel):
    pass
    
class ParserService:
    def process(self, input: ParserInput) -> ParserOutput:
        pass