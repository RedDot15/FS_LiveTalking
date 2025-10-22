from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import asynccontextmanager
from contextlib import contextmanager

import httpx

from .chat_service import LiteLLMChatService
from .embedding_service import LiteLLMEmbeddingService

from logger import get_logger
logger = get_logger(__name__)


class LiteLLMService(
    LiteLLMChatService,
    LiteLLMEmbeddingService
):
    """
    Service for interacting with LiteLLM API for language model inference and embedding generation.

    This service provides both synchronous and asynchronous methods for making
    requests to LiteLLM-compatible APIs, with support for various models including
    OpenAI and Claude models. It supports both text completion and embedding operations.
    """
    
    def __init__(self, url, model, embedding_model, frequency_penalty, n, presence_penalty, temperature, top_p, max_completion_tokens, encoding_format, dimensions, max_length):

        self.url = url
        self.model = model
        self.embedding_model = embedding_model
        self.frequency_penalty = frequency_penalty
        self.n = n
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.max_length = max_length

    @property
    @contextmanager
    def client(self) -> Generator[httpx.Client]:
        """
        Context manager for creating a synchronous HTTP client.

        Yields:
            httpx.Client: A configured HTTP client with authentication headers.

        Raises:
            ValueError: If authentication fails (401 status).
            httpx.HTTPStatusError: For other HTTP errors.
        """
        client = httpx.Client(
            base_url=str(self.url).rstrip('/'),
            headers={
                'Content-Type': 'application/json',
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        try:
            yield client
        except Exception as e:
            raise e
        finally:
            client.close()

    @property
    @asynccontextmanager
    async def async_client(self) -> AsyncGenerator[httpx.AsyncClient]:
        """
        Async context manager for creating an asynchronous HTTP client.

        Yields:
            httpx.AsyncClient: A configured async HTTP client with authentication headers.
        """
        client = httpx.AsyncClient(
            base_url=self.url.unicode_string().rstrip('/'),
            headers={
                'Content-Type': 'application/json',
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        yield client