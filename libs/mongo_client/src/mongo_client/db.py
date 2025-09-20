from typing import Any
from pymongo import MongoClient
from pymongo import ASCENDING, DESCENDING
from .settings import MongoSettings
from base import BaseService
from contextlib import contextmanager
from functools import cached_property
from typing import Generator
from pymongo.database import Database
 
class MongoDBHandler(BaseService):
    mongo_settings: MongoSettings
    
    @contextmanager
    def get_database(self) -> Generator[Database, None, None]:
        try:
            self.get_mongo_client
            yield self._db
        finally:
            self._client.close()

    @cached_property
    def get_mongo_client(self) -> MongoClient:
        # Construct the URI using the already-assigned mongo_settings attribute
        uri = f"mongodb://{self.mongo_settings.username}:{self.mongo_settings.password}@{self.mongo_settings.host}:{self.mongo_settings.port}/"
        
        # Initialize the client and database
        self._client = MongoClient(uri)
        self._db = self._client[self.mongo_settings.db]
        
        self.create_all_indexes()

        return self._client
    
    def create_all_indexes(self):
        """Creates all necessary indexes for the collections."""
        # Create an index on the `id` field for the 'characters' collection
        self._db.characters.create_index([("name", ASCENDING)], unique=True)
        
        # # Create indexes for the 'conversations' collection
        self._db.conversations.create_index([("participants_hash", ASCENDING),("created_at", DESCENDING)])

        # # Create indexes for the 'qa_pairs' collection
        self._db.qa_pairs.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])

    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used in MongoDBHandler.")

