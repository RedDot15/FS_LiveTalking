# Example of use:
import os
from dotenv import load_dotenv
from datetime import datetime
from mongo_client import (
    MongoSettings, 
    MongoDBHandler
)
from mongo_client.model import (
    Character,
    Conversation,
    QAPair
)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get credentials from environment variables
    db_name = os.getenv("MONGO__DB")
    user = os.getenv("MONGO__USER")
    password = os.getenv("MONGO__PASSWORD")
    host = os.getenv("MONGO__HOST")
    port = os.getenv("MONGO__PORT")

    db_handler = MongoDBHandler(
        mongo_settings = MongoSettings(
            db=db_name, 
            username=user, 
            password=password, 
            host=host, 
            port=port
        )
    )
    
    ################# Character #################
    # new_char = Character(name="AI Assistant", avatar_url="url_to_avatar")
    # print(db_handler.create_character(new_char))

    # print(db_handler.get_character())

    # print(db_handler.get_character_by_id(character_id="68b90225e6f59bb7290ad88e"))

    # new_char = Character(name="Nguyen", avatar_url="updated_url")
    # print(db_handler.update_character_by_id(character_id="68b90959b3c5183789417f6d", character=new_char))

    # print(db_handler.delete_character_by_id(character_id="68b9081ea7ea1b3baf1776b1"))

    ################# Conversation #################
    # new_convo = Conversation(name="General Inquiry", participants_hash="hash123", character_id="char_123", created_at=datetime.now())
    # print(db_handler.create_conversation(new_convo))

    # print(db_handler.get_conversation_by_participants_hash(participants_hash="hash123"))

    # print(db_handler.get_conversation_by_id(conversation_id="68b90225e6f59bb7290ad88f"))

    # print(db_handler.update_conversation_by_id(conversation_id="68b913a0a8aad2580c6fca7c", updated_conversation_name="Updated_name"))

    # print(db_handler.delete_conversation_by_id(conversation_id="68b90225e6f59bb7290ad88f"))

    ################# QA pair #################
    # new_qa = QAPair(conversation_id="convo_456", question="What's the capital of France?", answer="Paris", created_at=datetime.now(), updated_at=datetime.now(), response_time=500)
    # print(db_handler.create_qa_pair(new_qa))

    # print(db_handler.get_qa_pair_by_conversation_id("convo_456"))

    # print(db_handler.get_k_most_recent_qa_pair_by_conversation_id("convo_456", 3))

    # updated_qa_pair = QAPair(question="Updated question", answer="Updated answer", response_time=300)
    # print(db_handler.update_qa_pair_by_id(qa_pair_id="68b92dc2ad7283ee6ba67668", qa_pair=updated_qa_pair))

    # print(db_handler.delete_qa_pair_by_id(qa_pair_id="68b90226e6f59bb7290ad890"))

    db_handler.close()