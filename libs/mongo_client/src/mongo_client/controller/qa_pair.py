from datetime import datetime
from ..model import QAPair
from bson.objectid import ObjectId

class QAPairHandler:

    # Create new qa_pair
    def create_qa_pair(self, qa_pair: QAPair):
        collection = self._get_collection("qa_pairs")
        result = collection.insert_one(qa_pair.__dict__)
        return result.inserted_id

    # Get all qa_pairs by conversation_id with (created_at order: asc)
    def get_qa_pair_by_conversation_id(self, conversation_id: str):
        collection = self._get_collection("qa_pairs")
        data = collection.find({"conversation_id": conversation_id}).sort("created_at", 1)
        return list(data)
    
    # Get 3 most recent qa_pair by conversation_id with (created_at order: desc)
    def get_3_most_recent_qa_pair_by_conversation_id(self, conversation_id: str):
        collection = self._get_collection("qa_pairs")
        data = collection.find({"conversation_id": conversation_id}).sort("created_at", -1).limit(3)
        return list(data)
    
    # Update qa_pair by id ( can only update (question,answer,response_time) )
    def update_qa_pair_by_id(self, qa_pair_id: str, qa_pair: QAPair):
        collection = self._get_collection("qa_pairs")
        update_data = {
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "response_time": qa_pair.response_time,
            "updated_at": datetime.now()
        }
        return collection.update_one(
            {"_id": ObjectId(qa_pair_id)},
            {"$set": update_data}
        )
    
    # Delete qa_pair_by_id
    def delete_qa_pair_by_id(self, qa_pair_id: str):
        collection = self._get_collection("qa_pairs")
        return collection.delete_one({"_id": ObjectId(qa_pair_id)})