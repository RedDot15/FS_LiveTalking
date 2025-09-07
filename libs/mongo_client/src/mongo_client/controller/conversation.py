from ..model import Conversation
from bson.objectid import ObjectId

class ConversationHandler:

    # Create new conversation
    def create_conversation(self, conversation: Conversation):
        collection = self._get_collection("conversations")
        result = collection.insert_one(conversation.__dict__)
        return result.inserted_id

    # Get all conversation by participants_hash with (created_at order: desc)
    def get_conversation_by_participants_hash(self, participants_hash: str):
        collection = self._get_collection("conversations")
        data = collection.find({"participants_hash": participants_hash}).sort("created_at", 1)
        return list(data)

    # Get conversation by id
    def get_conversation_by_id(self, conversation_id: str):
        collection = self._get_collection("conversations")
        data = collection.find_one({"_id": ObjectId(conversation_id)})
        return data
    
    # Update conversation by id (can only update name)
    def update_conversation_by_id(self, conversation_id: str, updated_conversation_name: str):
        collection = self._get_collection("conversations")
        update_data = {
            "name": updated_conversation_name,
        }
        return collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": update_data}
        )
    
    # Delete conversation by id
    def delete_conversation_by_id(self, conversation_id: str):
        collection = self._get_collection("conversations")
        return collection.delete_one({"_id": ObjectId(conversation_id)})