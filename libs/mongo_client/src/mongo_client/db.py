from __future__ import annotations

from typing import Any, Generator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from .settings import MongoSettings
from base import BaseService
from contextlib import contextmanager
from bson.binary import UuidRepresentation

class MongoDBHandler:

    def __init__(self, db: str, username: str, password: str, host: str, port: int):
        """
        Initialize the handler, establish a connection to MongoDB, and create indexes.
        The connection is established only once when the object is created.
        """
        self.db = db
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        
        uri = (
            f"mongodb://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/"
            f"{self.db}?authSource=admin"
        )
        
        self._client = MongoClient(uri)
        self._db = self._client[self.db]

        self._create_all_indexes()

    def _create_all_indexes(self):
        """Create all necessary indexes for the collections."""
        if self._db is None:
            return

        self._db.characters.create_index([("name", ASCENDING)], unique=True)
        self._db.conversations.create_index([("participants_hash", ASCENDING), ("created_at", DESCENDING)])
        self._db.qa_pairs.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])
        self._db.requests.create_index([("created_at", DESCENDING)])
        self._db.requests.create_index([("character_name", ASCENDING)])
        self._db.ratings.create_index([("character_id", DESCENDING), ("created_at", DESCENDING)])
        self._db.ratings.create_index([("character_id", DESCENDING), ("commented_by", DESCENDING)])

    @contextmanager
    def get_database(self) -> Generator[Database, None, None]:
        """
        Context manager to provide the database object.
        The connection is already established in __init__.
        """
        try:
            yield self._db
        finally:
            pass

    def close_connection(self):
        """
        This method will be called by the lifespan to close the connection when the application shuts down.
        """
        if self._client:
            self._client.close()
            
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used in MongoDBHandler.")