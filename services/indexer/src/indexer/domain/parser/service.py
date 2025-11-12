from __future__ import annotations

from base import BaseModel, BaseService
from fastapi import UploadFile
from logger import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm, pymupdf
import asyncio
import os
import shutil
from litellm import LiteLLMEmbeddingInput, LiteLLMService


class ParserInput(BaseModel):
    knowledge_file_local_path: str
    
class ParserOutput(BaseModel):
    chunks: list[str]
    embeddings: list
    
class ParserService(BaseService):
    litellm: LiteLLMService

    async def extract_text_from_pdf(self, inputs: ParserInput):
        knowledge_file_local_path = inputs.knowledge_file_local_path

        if knowledge_file_local_path.endswith('pdf'):
            try:
                md_text = pymupdf4llm.to_markdown(knowledge_file_local_path)
                await knowledge_file.seek(0)    
                os.remove(knowledge_file_local_path)
                return md_text
            except Exception as e:
                os.remove(knowledge_file_local_path)
                raise e
        elif knowledge_file_local_path.endswith('txt'):
            try:
                with open(knowledge_file_local_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()
                os.remove(knowledge_file_path)
                return md_text
            except Exception as e:
                os.remove(knowledge_file_path)
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