from __future__ import annotations

from ..utils import get_settings

from typing import Any
import asyncio
import re

from base import BaseModel
from realistic import LipReal  # Giả định BaseReal được import từ đây hoặc nơi khác
from llm_client import LLMService, LLMServiceInput
from llm_client import Message
from llm_client import MessageRole
from llm_client import CompletionMessage

from logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def llm_response(nerfreal: LipReal, text: str):
    logger.info("Initializing LLM service and sending request...")
    llm = LLMService(settings=settings.llm)
    
    response = await llm.aprocess(
        input=LLMServiceInput(
            message=[CompletionMessage(
                content=text,
                role=MessageRole.USER)
            ],
            model=settings.llm.model_name,
        )
    )
    logger.info("Received full response from LLM.")

    full_text = ""
    if isinstance(response.response, str):
        full_text = response.response
    elif isinstance(response.response, BaseModel):
        if hasattr(response.response, 'content'):
            full_text = response.response.content
        elif hasattr(response.response, 'text'):
            full_text = response.response.text

    if not full_text:
        logger.warning("LLM did not return any text content to process.")
        return

    sentences = re.split(r'(?<=[,.!?;:，。！？：；])\s*', full_text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if not sentences:
        sentences = [full_text]

    logger.info(f"Split into {len(sentences)} sentences. Starting to send to nerfreal...")
    for i, sentence in enumerate(sentences):
        if sentence:
            logger.info(f"Sending sentence {i+1}/{len(sentences)}: '{sentence}'")
            nerfreal.put_msg_txt(sentence)

            await asyncio.sleep(0.2)

    logger.info("All sentences sent to nerfreal successfully.")