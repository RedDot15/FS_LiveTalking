from __future__ import annotations

from base import BaseModel, BaseService
from fastapi import UploadFile
from logger import get_logger
from langchain.text_splitter import TextSplitter
import pymupdf4llm, pymupdf

class ParserInput(BaseModel):
    knowledge_file: UploadFile
    
class ParserOutput(BaseModel):
    parsed_text: str
    
class ParserService(BaseService):
    async def process(self, inputs: ParserInput) -> ParserOutput:
        knowledge_file = inputs.knowledge_file
        if knowledge_file.filename.endswith('pdf'):
            try:
                md_text = pymupdf4llm.to_markdown(knowledge_file)
            except Exception as e:
                raise e
        return ParserOutput(
            parsed_text=md_text
        )
        