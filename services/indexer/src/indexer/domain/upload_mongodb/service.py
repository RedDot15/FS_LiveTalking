from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
from pydantic import dataclasses
import os
import shutil
from mongo_client import MongoDBHandler, MongoSettings
from mongo_client.model import Character

class CharacterMongoDBInputs(BaseModel):
    # character_id: str
    name: str
    avatar_url: str

class CharacterMongoDBOutputs(BaseModel):
    result: str

class CharacterUploadMongoDBService(BaseService):
    db_handler: MongoDBHandler
    async def upload_to_mongo(self, inputs: CharacterMongoDBInputs):
        new_character = Character(
            name = inputs.name,
            avatar_url = inputs.avatar_url,
        )
        try:
            result = self.db_handler.create_character(new_character)
        except Exception as e:
            raise e
        return CharacterMongoDBOutputs(result=result.inserted_id)
    
    async def process(self, inputs):
        return super().process(inputs)