from bson.objectid import ObjectId
from ..model import Character

class CharacterHandler:

    # Create a new character
    def create_character(self, character: Character):
        collection = self._get_collection("characters")
        result = collection.insert_one(character.__dict__)
        return result

    # Get all characters with (name order: asc)
    def get_character(self):
        collection = self._get_collection("characters")
        data = collection.find().sort("name", 1)
        return list(data)

    # Get character by id
    def get_character_by_id(self, character_id: str):
        collection = self._get_collection("characters")
        data = collection.find_one({"_id": ObjectId(character_id)})
        return data
    
    # Update character by id
    def update_character_by_id(self, character_id: str, character: Character):
        collection = self._get_collection("characters")
        update_data = character.__dict__
        return collection.update_one(
            {"_id": ObjectId(character_id)},
            {"$set": update_data}
        )

    # Delete character by id
    def delete_character_by_id(self, character_id: str):
        collection = self._get_collection("characters")
        return collection.delete_one({"_id": ObjectId(character_id)})