from datetime import datetime
from typing import Any
from ..model import QAPair
from bson.objectid import ObjectId
from base import BaseService
from pymongo.collection import Collection

class QAPairHandler(BaseService):
    collection: Collection

    # Create new qa_pair
    def create_qa_pair(self, qa_pair: QAPair):
        result = self.collection.insert_one(qa_pair.__dict__)
        return result

    # Get all qa_pairs by conversation_id with (created_at order: asc)
    def get_qa_pairs_by_conversation_id(self, conversation_id: str):
        data = self.collection.find({"conversation_id": conversation_id}).sort("created_at", 1)
        return list(data)
    
    # Get 3 most recent qa_pair by conversation_id with (created_at order: desc)
    def get_k_most_recent_qa_pair_by_conversation_id(self, conversation_id: str, k: int):
        data = self.collection.find({"conversation_id": conversation_id}).sort("created_at", -1).limit(k)
        return list(data)
    
    # Update qa_pair by id ( can only update (question,answer,response_time) )
    def update_qa_pair_by_id(self, qa_pair_id: str, qa_pair: QAPair):
        update_data = {
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "response_time": qa_pair.response_time,
            "updated_at": datetime.now()
        }
        return self.collection.update_one(
            {"_id": ObjectId(qa_pair_id)},
            {"$set": update_data}
        )
    
    # Delete qa_pair_by_id
    def delete_qa_pair_by_id(self, qa_pair_id: str):
        return self.collection.delete_one({"_id": ObjectId(qa_pair_id)})
    
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used.")