from __future__ import annotations
from datetime import datetime
from uuid import uuid4

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from mongo_client.controller.rating import RatingHandler
from mongo_client.model.entity import Rating
from pydantic import ConfigDict
from pydantic import Field

class RatingServiceOutput(BaseModel):
    ratings: list[dict]

class AddRatingInput(BaseModel):
    character_id: str
    rating: float
    comment: str

class AddRatingOutput(BaseModel):
    rating: dict

class UpdateRatingInput(BaseModel):
    rating_id: str
    rating: float
    comment: str

class UpdateRatingOutput(BaseModel):
    rating: dict

class RatingServiceApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]

    def process(self, character_id: str) -> RatingServiceOutput:
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                rating_handler = RatingHandler(collection=mongodb["ratings"])
                ratings = rating_handler.get_rating_by_character_id(character_id=character_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
            
        return RatingServiceOutput(ratings=ratings)
    
    def add_rating(self, input: AddRatingInput, current_user_id: str, current_username: str):
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                rating_handler = RatingHandler(collection=mongodb["ratings"])

                rating = rating_handler.get_rating_by_character_id_and_user_id(character_id=input.character_id, user_id=current_user_id)
                if rating:
                    raise Exception(f"Rating already exists.")
                
                rating_id = str(uuid4())
                rating_handler.create_rating(Rating(
                    _id = rating_id,
                    character_id = input.character_id,
                    commented_by = {
                        "id": current_user_id,
                        "username": current_username
                    },
                    rating = input.rating,
                    comment = input.comment,
                    created_at = datetime.now(),
                    updated_at = datetime.now()
                ))
                # Fetch the created rating to return
                rating = rating_handler.get_rating_by_id(rating_id=rating_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
            
        return AddRatingOutput(
            rating = rating
        )
        
        
    def update_rating(self, input: UpdateRatingInput, current_user_id: str, current_username: str):
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                rating_handler = RatingHandler(collection=mongodb["ratings"])
                rating = rating_handler.get_rating_by_id(rating_id = input.rating_id)

                if not rating:
                    raise Exception(f"Rating not found.")
                if rating.commented_by.id != current_user_id or rating.commented_by.username != current_username:
                    raise Exception("Unauthorized user")

                rating.rating = input.rating
                rating.comment = input.comment
                rating.updated_at = datetime.now()
                rating = rating_handler.update_rating_by_id(rating_id=input.rating_id, rating = rating)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
            
        return UpdateRatingOutput(
            rating = rating
        )

    def delete_rating(self, rating_id: str, current_user_id: str, current_username: str):
        
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                # Get character
                rating_handler = RatingHandler(collection=mongodb["ratings"])
                rating = rating_handler.get_rating_by_id(rating_id=rating_id)

                if not rating:
                    raise Exception("Rating not found")
                # Validate rating owner
                if rating.commented_by.id != current_user_id or rating.commented_by.username != current_username:
                    raise Exception(f"Unauthorize user: {current_user_id}")

                rating_handler.delete_rating_by_id(rating_id=rating_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        return