import os
from dotenv import load_dotenv
from mongo_client.controller import CharacterHandler, ConversationHandler, QAPairHandler 
from pymongo import MongoClient
from pymongo import ASCENDING, DESCENDING

class MongoDBHandler(CharacterHandler, ConversationHandler, QAPairHandler):
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get credentials from environment variables
        db_name = os.getenv("MONGO__DB")
        user = os.getenv("MONGO__USER")
        password = os.getenv("MONGO__PASSWORD")
        host = os.getenv("MONGO__HOST")
        port = os.getenv("MONGO__PORT")

        # Construct the URI using the retrieved variables
        uri = f"mongodb://{user}:{password}@{host}:{port}/"
        
        # Initialize the client with the dynamic URI
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        self.create_all_indexes()

    def _get_collection(self, collection_name: str):
        return self.db[collection_name]

    def create_all_indexes(self):
        """Creates all necessary indexes for the collections."""
        # Create an index on the `id` field for the 'characters' collection
        self.db.characters.create_index([("name", ASCENDING)], unique=True)
        
        # # Create indexes for the 'conversations' collection
        self.db.conversations.create_index([("participants_hash", ASCENDING),("created_at", DESCENDING)])

        # # Create indexes for the 'qa_pairs' collection
        self.db.qa_pairs.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])

    def close(self):
        self.client.close()

