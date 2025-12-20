from typing import Any
from ..model import Conversation
from base import BaseService
from pymongo.collection import Collection

class ConversationHandler(BaseService):
    collection: Collection

    # Create new conversation
    def create_conversation(self, conversation: Conversation):
        result = self.collection.insert_one(conversation.__dict__)
        return result

    # Get all conversation by participants_hash with (created_at order: desc)
    def get_conversation_by_participants_hash(self, participants_hash: str):
        query = {
            "participants_hash": participants_hash,
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find(query).sort("created_at", -1)
        return list(data)

    # Get conversation by id
    def get_conversation_by_id(self, conversation_id: str):
        query = {
            "_id": conversation_id,
            "is_deleted": {"$ne": True}
        }
        data = self.collection.find_one(query)
        return data
    
    # Update conversation by id (can only update name)
    def update_conversation_by_id(self, conversation_id: str, updated_conversation_name: str):
        update_data = {
            "name": updated_conversation_name,
        }
        return self.collection.update_one(
            {"_id": conversation_id, "is_deleted": {"$ne": True}},
            {"$set": update_data}
        )
    
    # Delete conversation by id
    def delete_conversation_by_id(self, conversation_id: str):
        return self.collection.update_one(
            {"_id": conversation_id},
            {"$set": {"is_deleted": True}}
        )
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")