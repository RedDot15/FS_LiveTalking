from typing import Any
from base import BaseService
from mongo_client.model.entity import Rating
from pymongo.collection import Collection

class RatingHandler(BaseService):
    collection: Collection

    # Create a new character
    def create_rating(self, rating: Rating):
        result = self.collection.insert_one(rating.__dict__)
        return result

    # Get all ratings with (name order: asc)
    def get_rating_by_character_id(self, character_id: str):
        data = self.collection.find({"character_id": character_id}).sort("created_at", 1)
        return list(data)

    # Get character by id
    def get_rating_by_id(self, rating_id: str):
        data = self.collection.find_one({"_id": rating_id})
        return data
    
    def get_rating_by_character_id_and_user_id(self, character_id: str, user_id: str):
        data = self.collection.find_one({"character_id": character_id, "user_id": user_id})
        return data

    # Update character by id
    def update_rating_by_id(self, rating_id: str, rating: Rating):
        update_data = rating.__dict__
        return self.collection.update_one(
            {"_id": rating_id},
            {"$set": update_data}
        )

    # Delete character by id
    def delete_rating_by_id(self, rating_id: str):
        return self.collection.delete_one({"_id": rating_id})
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")