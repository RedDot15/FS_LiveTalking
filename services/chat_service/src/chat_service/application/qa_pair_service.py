from __future__ import annotations

from typing import Annotated, Any

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field

from mongo_client.controller import QAPairHandler

logger = get_logger(__name__)
class QAPairInput(BaseModel):
    conversation_id: str
    
    
class QAPairOutput(BaseModel):
    qa_pairs: list[dict]
    
    
class QAPairService(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    async def process(self, input: QAPairInput) -> QAPairOutput:
        
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                qa_pair_handler = QAPairHandler(collection=mongodb["qa_pairs"])
                conversations = qa_pair_handler.get_qa_pairs_by_conversation_id(conversation_id=input.conversation_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        return QAPairOutput(qa_pairs=conversations)