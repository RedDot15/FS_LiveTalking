from __future__ import annotations
from fastapi import UploadFile, File, Form

from base import BaseModel, BaseService
from pydantic import dataclasses
import os
import shutil
from mongo_client import MongoDBHandler, MongoSettings
from mongo_client.model import Character
from mongo_client.controller import CharacterHandler

class CharacterMongoDBInputs(BaseModel):
    character_id: str
    name: str
    created_by: str

class CharacterMongoDBOutputs(BaseModel):
    result: str

class CharacterUploadMongoDBService(BaseService):
    db_handler: MongoDBHandler
    async def process(self, inputs: CharacterMongoDBInputs):
        new_character = Character(
            name = inputs.name,
            _id = inputs.character_id,
            created_by = inputs.created_by
        )
        try:
            with self.db_handler.get_database() as db:
                char_handler = CharacterHandler(collection=db["characters"])
                result = char_handler.create_character(new_character)
        except Exception as e:
            raise e
        return CharacterMongoDBOutputs(result=str(result))