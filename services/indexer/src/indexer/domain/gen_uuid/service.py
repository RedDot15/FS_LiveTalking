from __future__ import annotations

from base import BaseService, BaseModel
from indexer.shared.tools import generate_index_id

class CharacterIdInputs(BaseModel):
    name: str

class CharacterIdOutputs(BaseModel):
    character_id: str

class GenUUIDService(BaseService):
    async def process(self, inputs: CharacterIdInputs) -> CharacterIdOutputs:
        character_id = await generate_index_id(
            character_name=inputs.name
        )
        return CharacterIdOutputs(
            character_id=character_id
        )