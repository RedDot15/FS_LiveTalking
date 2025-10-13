from __future__ import annotations

from base import BaseModel, BaseService
from fastapi import UploadFile
from logger import get_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymupdf4llm, pymupdf
import asyncio
import os
import shutil
from litellm import LiteLLMEmbeddingInput, LiteLLMService


class ParserInput(BaseModel):
    knowledge_file: UploadFile
    
class ParserOutput(BaseModel):
    chunks: list[str]
    embeddings: list
    
class ParserService(BaseService):
    litellm: LiteLLMService

    async def extract_text_from_pdf(self, inputs: ParserInput):
        knowledge_file = inputs.knowledge_file
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)

        # Đường dẫn file tạm
        knowledge_file_name = knowledge_file.filename
        temp_file_path = os.path.join(temp_dir, knowledge_file_name)

        # Lưu file người dùng upload vào thư mục tạm
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(knowledge_file.file, buffer)
        if temp_file_path.endswith('pdf'):
            try:
                md_text = pymupdf4llm.to_markdown(temp_file_path)
                await knowledge_file.seek(0)    
                os.remove(temp_file_path)
                os.removedirs(temp_dir)
                return md_text
            except Exception as e:
                raise e

    
    async def split_text_into_chunks(self, text) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    async def embed_chunk(self, chunks: list[str]):
        embed_chunk: list = []

        with self.litellm.client as client:
            for chunk in chunks:
                result = self.litellm.embedding(client=client, inputs=LiteLLMEmbeddingInput(
                    input=chunk,
                    encoding_format=self.litellm.encoding_format,
                    count_tokens=True,
                    embedding_model=self.litellm.embedding_model,
                    dimensions=self.litellm.dimensions
                ))
                embed_chunk.append(result.vector)
        return embed_chunk
    
    async def process(self, inputs: ParserInput) -> ParserOutput:
        
        text_content = await self.extract_text_from_pdf(inputs=inputs)

        chunks = await self.split_text_into_chunks(text_content)

        embeddings = await self.embed_chunk(chunks=chunks)

        # print(f"Embedding size:{embedding.shape}")
        
        return ParserOutput(
            chunks=chunks,
            embeddings=embeddings,
        )