from typing import Any
from mongo_client.controller import CharacterHandler, ConversationHandler, QAPairHandler 
from pymongo import MongoClient
from pymongo import ASCENDING, DESCENDING
from .settings import MongoSettings
from base import BaseService

class MongoDBHandler(CharacterHandler, ConversationHandler, QAPairHandler, BaseService):
    mongo_settings: MongoSettings

    def model_post_init(self, __context: Any) -> None:
        """
        This method runs after the BaseModel has been initialized.
        """
        # Construct the URI using the already-assigned mongo_settings attribute
        uri = f"mongodb://{self.mongo_settings.username}:{self.mongo_settings.password}@{self.mongo_settings.host}:{self.mongo_settings.port}/"
        
        # Initialize the client and database
        self._client = MongoClient(uri)
        self._db = self._client[self.mongo_settings.db]
        self.create_all_indexes()

    def _get_collection(self, collection_name: str):
        return self._db[collection_name]

    def create_all_indexes(self):
        """Creates all necessary indexes for the collections."""
        # Create an index on the `id` field for the 'characters' collection
        self._db.characters.create_index([("name", ASCENDING)], unique=True)
        
        # # Create indexes for the 'conversations' collection
        self._db.conversations.create_index([("participants_hash", ASCENDING),("created_at", DESCENDING)])

        # # Create indexes for the 'qa_pairs' collection
        self._db.qa_pairs.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])

    def close(self):
        self._client.close()

    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used in MongoDBHandler.")

