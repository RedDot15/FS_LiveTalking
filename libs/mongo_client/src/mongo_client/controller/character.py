from typing import Any
from base import BaseService
from bson.objectid import ObjectId
from ..model import Character
from pymongo.collection import Collection

class CharacterHandler(BaseService):
    collection: Collection

    # Create a new character
    def create_character(self, character: Character):
        result = self.collection.insert_one(character.__dict__)
        return result

    # Get all characters with (name order: asc)
    def get_character(self):
        data = self.collection.find().sort("name", 1)
        return list(data)

    # Get character by id
    def get_character_by_id(self, character_id: str):
        data = self.collection.find_one({"_id": ObjectId(character_id)})
        return data
    
    # Update character by id
    def update_character_by_id(self, character_id: str, character: Character):
        update_data = character.__dict__
        return self.collection.update_one(
            {"_id": ObjectId(character_id)},
            {"$set": update_data}
        )

    # Delete character by id
    def delete_character_by_id(self, character_id: str):
        return self.collection.delete_one({"_id": ObjectId(character_id)})
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")