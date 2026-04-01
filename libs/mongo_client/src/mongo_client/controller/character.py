from typing import Any
from base import BaseService
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
        query = {"is_deleted": {"$ne": True}}
        data = self.collection.find(query).sort("name", 1)
        return list(data)

    # Get all characters with (name order: asc) by created_by
    def get_character_by_created(self, created_by: str):
        query = {"created_by": created_by, "is_deleted": {"$ne": True}}
        data = self.collection.find(query).sort("name", 1)
        return list(data)

    # Get character by id
    def get_character_by_id(self, character_id: str):
        query = {
            "_id": character_id, 
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find_one(query)
        return data
    
    # Get character by id
    # def get_character_by_name(self, character_name: str):
    #     data = self.collection.find_one({"name": character_name})
    #     return data
    
    # Update character by id
    # def update_character_by_id(self, character_id: str, character: Character):
    #     update_data = character.__dict__
    #     return self.collection.update_one(
    #         {"_id": character_id},
    #         {"$set": update_data}
    #     )

    # Delete character by id
    def delete_character_by_id(self, character_id: str):
        return self.collection.update_one(
            {"_id": character_id},
            {"$set": {"is_deleted": True}}
        )
    
    def delete_characters_by_created_by(self, created_by: str):
        return self.collection.update_many(
            {"created_by": created_by},
            {"$set": {"is_deleted": True}}
        )
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")