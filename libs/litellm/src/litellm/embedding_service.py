from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Generator
from typing import Any
from typing import Dict
from typing import Optional

import httpx

from .datatypes import TokensLLM

from base import BaseModel

class LiteLLMEmbeddingInput(BaseModel):
    """
    Input model for LiteLLM embedding requests.

    Attributes:
        input (str | list[str]): The text(s) to embed.
        embedding_model (str): The model name to use for embedding.
        encoding_format (Optional[str]): The format to return the embeddings in.
        count_tokens (bool): Whether to count tokens in the response.
    """

    input: str | list[str]
    embedding_model: str
    encoding_format: Optional[str] = None
    dimensions: int
    count_tokens: bool = False

class LiteLLMEmbeddingOutput(BaseModel):
    """
    Output model for LiteLLM embedding responses.

    Attributes:
        vector (list): The embedding vector returned from the API.
    """

    vector: list
    
class LiteLLMEmbeddingService:
    
    def embedding(self, client: Generator[httpx.Client], inputs: LiteLLMEmbeddingInput) -> LiteLLMEmbeddingOutput:
        """
        Generate embeddings for the given input text(s).

        Args:
            inputs (LiteLLMEmbeddingInput): Input parameters for the embedding request.

        Returns:
            LiteLLMEmbeddingOutput: The processed embedding output.
        """
        return self.__embedding_by_llm(
            client=client,
            input=inputs.input,
            embedding_model=inputs.embedding_model,
            encoding_format=inputs.encoding_format,
            count_tokens=inputs.count_tokens,
            dimensions=inputs.dimensions,
        )

    async def embedding_async(
        self,
        inputs: LiteLLMEmbeddingInput,
    ) -> LiteLLMEmbeddingOutput:
        """
        Generate embeddings for the given input text(s) asynchronously.

        Args:
            inputs (LiteLLMEmbeddingInput): Input parameters for the embedding request.

        Returns:
            LiteLLMEmbeddingOutput: The processed embedding output.
        """
        return await self.__embedding_by_llm_async(
            input=inputs.input,
            embedding_model=inputs.embedding_model,
            encoding_format=inputs.encoding_format,
            count_tokens=inputs.count_tokens,
            dimensions=inputs.dimensions,
        )
        
    def __embedding_by_llm(
        self,
        *,
        client: Generator[httpx.Client],
        input: str | list[str],
        embedding_model: str,
        encoding_format: str | None,
        count_tokens: bool = False,
        dimensions: int,
    ) -> LiteLLMEmbeddingOutput:
        """
        Execute synchronous embedding generation using the LLM API.

        Args:
            input (str | list[str]): The text(s) to embed.
            embedding_model (str): The model name to use for embedding.
            encoding_format (str | None): The format to return the embeddings in.
            count_tokens (bool): Whether to count tokens in the response.

        Returns:
            LiteLLMEmbeddingOutput: The processed embedding response.

        Raises:
            httpx.HTTPStatusError: For HTTP-related errors.
            Exception: For other unexpected errors.
        """
        with self.client as client:
            payload = self.__build_embedding_payload(
                input=input,
                embedding_model=embedding_model,
                encoding_format=encoding_format,
                dimensions=dimensions,
            )
            try:
                response = client.post('/v1/embeddings', json=payload)
                response.raise_for_status()
                response_data = response.json()

                return self.__postprocessing_embedding_response(
                    response=response_data,
                    count_token=count_tokens,
                )

            except Exception as e:
                raise e

    async def __embedding_by_llm_async(
        self,
        *,
        client: AsyncGenerator[httpx.AsyncClient],
        input: str | list[str],
        embedding_model: str,
        encoding_format: str | None,
        count_tokens: bool = False,
        dimensions: int,
    ) -> LiteLLMEmbeddingOutput:
        """
        Execute asynchronous embedding generation using the LLM API.

        Args:
            input (str | list[str]): The text(s) to embed.
            embedding_model (str): The model name to use for embedding.
            encoding_format (str | None): The format to return the embeddings in.
            count_tokens (bool): Whether to count tokens in the response.

        Returns:
            LiteLLMEmbeddingOutput: The processed embedding response.

        Raises:
            httpx.HTTPStatusError: For HTTP-related errors.
            Exception: For other unexpected errors.
        """
        payload = self.__build_embedding_payload(
            input=input,
            embedding_model=embedding_model,
            encoding_format=encoding_format,
            dimensions=dimensions,
        )

        try:
            response = await client.post('/v1/embeddings', json=payload)
            response.raise_for_status()
            response_data = response.json()

            return self.__postprocessing_embedding_response(
                response=response_data,
                count_token=count_tokens,
            )

        except Exception as e:
            raise e
            
    def __build_embedding_payload(
        self,
        input: str | list[str],
        embedding_model: str,
        encoding_format: str | None,
        dimensions: int,
    ) -> Dict[str, Any]:
        """
        Build the request payload for the embedding API.

        Args:
            input (str | list[str]): The text(s) to embed.
            embedding_model (str): The model name to use for embedding.
            encoding_format (str | None): The format to return the embeddings in.

        Returns:
            Dict[str, Any]: The formatted request payload for the API.
        """
        payload = {
            'input': input,
            'model': embedding_model,
            'dimensions': dimensions,
        }

        if encoding_format:
            payload['encoding_format'] = encoding_format

        return payload

    def __postprocessing_embedding_response(
        self,
        response: Dict[str, Any],
        count_token: bool,
    ) -> LiteLLMEmbeddingOutput:
        """
        Post-process the response from embedding API.

        Args:
            response (Dict[str, Any]): The response object received from the embedding API.
            count_token (bool): Flag indicating whether to count tokens used in the response (currently unused).

        Returns:
            LiteLLMEmbeddingOutput: The processed output containing the first embedding vector.

        Raises:
            ValueError: If the response data is empty or invalid.
        """
        if not response.get('data'):
            raise ValueError('No data returned in embedding response')

        embeddings = [item['embedding'] for item in response['data']]

        tokens = TokensLLM()
        if count_token and response.get('usage'):
            usage = response['usage']
            tokens.prompt_tokens = usage.get('prompt_tokens', 0)
            tokens.total_tokens = usage.get('total_tokens', 0)

        return LiteLLMEmbeddingOutput(
            vector=embeddings[0],
        )