from typing import Any
from base import BaseService
from mongo_client.model.entity import Rating
from pymongo.collection import Collection

class RatingHandler(BaseService):
    collection: Collection

    # Create a new character
    def create_rating(self, rating: Rating):
        rating.is_deleted = False 
        result = self.collection.insert_one(rating.__dict__)
        return result

    # Get all ratings with (name order: asc)
    def get_rating_by_character_id(self, character_id: str):
        query = {
            "character_id": character_id, 
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find(query).sort("created_at", -1)
        return list(data)

    # Get character by id
    def get_rating_by_id(self, rating_id: str):
        query = {
            "_id": rating_id, 
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find_one(query)
        return data
    
    def get_rating_by_character_id_and_user_id(self, character_id: str, user_id: str):
        query = {
            "character_id": character_id, 
            "commented_by.id": user_id, 
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find_one(query)
        return data

    # Update character by id
    def update_rating_by_id(self, rating_id: str, rating: Rating):
        update_data = rating
        return self.collection.update_one(
            {"_id": rating_id, "is_deleted": {"$ne": True}}, # Only update if not deleted
            {"$set": update_data}
        )

    # Delete character by id
    def delete_rating_by_id(self, rating_id: str):
        return self.collection.update_one(
            {"_id": rating_id},
            {"$set": {"is_deleted": True}}
        )

    def delete_ratings_by_created_by(self, created_by: str):
        return self.collection.update_many(
            {
                "commented_by.id": created_by,
                "is_deleted": {"$ne": True} 
            },
            {
                "$set": {"is_deleted": True}
            }
        )
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")