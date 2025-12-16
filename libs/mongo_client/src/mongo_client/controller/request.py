from typing import Any
from base import BaseService
from ..model import Request
from pymongo.collection import Collection

class RequestHandler(BaseService):
    collection: Collection

    # Create a new request
    def create_request(self, request: Request):
        result = self.collection.insert_one(request.__dict__)
        return result

    # Get all requests with (name order: asc)
    def get_requests(self):
        query = {"is_deleted": {"$ne": True}}
        data = self.collection.find(query).sort("created_at", -1)
        return list(data)

    # Get request by id
    def get_request_by_id(self, request_id: str):
        query = {
            "_id": request_id,
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find_one(query)
        return data
    
    # Get request by id
    # def get_request_by_character_name(self, character_name: str):
    #     data = self.collection.find_one({"character_name": character_name})
    #     return data
    
    # Update request by id
    def update_request_by_id(self, request_id: str, request: Request):
        return self.collection.update_one(
            {"_id": request_id},
            {"$set": request}
        )

    # Delete request by id
    def delete_request_by_id(self, request_id: str):
        return self.collection.update_one(
            {"_id": request_id},
            {"$set": {"is_deleted": True}}
        )

    # Soft Delete all requests by created_by (User ID)
    def delete_requests_by_created_by(self, created_by: str):
        return self.collection.update_many(
            {"created_by": created_by},
            {"$set": {"is_deleted": True}}
        )
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")