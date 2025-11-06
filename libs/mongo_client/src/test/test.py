# Example of use:
import os
from dotenv import load_dotenv
from datetime import datetime
from mongo_client import (
    MongoSettings, 
    MongoDBHandler
)
from mongo_client.controller import (
    CharacterHandler,
    ConversationHandler,
    QAPairHandler
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
    host = 'localhost'
    port = '27018'

    db_handler = MongoDBHandler(
        mongo_settings = MongoSettings(
            db=db_name, 
            username=user, 
            password=password, 
            host=host, 
            port=port
        )
    )
    
    with db_handler.get_database() as db:
        ################# Character #################
        # char_handler = CharacterHandler(collection=db["characters"])

        # new_char = Character(_id="410d4fa6-72e8-472e-b2cc-1fb5c9147d45", name="AI Assistant 4")
        # print(char_handler.create_character(new_char))

        # print(char_handler.get_character())

        print(char_handler.get_character_by_id(character_id="932d0f5a-30f3-4296-9fdb-eba73ef5695f"))

        # new_char = Character(name="Nguyen", avatar_url="updated_url")
        # print(char_handler.update_character_by_id(character_id="68b90959b3c5183789417f6d", character=new_char))

        # print(char_handler.delete_character_by_id(character_id="68b9081ea7ea1b3baf1776b1"))

        ################# Conversation #################
        # convo_handler = ConversationHandler(collection=db["conversations"])

        # new_convo = Conversation(name="General Inquiry", participants_hash="hash123", character_id="char_123", created_at=datetime.now())
        # print(convo_handler.create_conversation(new_convo))

        # print(convo_handler.get_conversation_by_participants_hash(participants_hash="hash123"))

        # print(convo_handler.get_conversation_by_id(conversation_id="68b90225e6f59bb7290ad88f"))

        # print(convo_handler.update_conversation_by_id(conversation_id="68b913a0a8aad2580c6fca7c", updated_conversation_name="Updated_name"))

        # print(convo_handler.delete_conversation_by_id(conversation_id="68b90225e6f59bb7290ad88f"))

        ################# QA pair #################
        # qa_handler = QAPairHandler(collection=db["qa_pairs"])

        # new_qa = QAPair(_id="e9949653-50ac-46ff-b943-66d0be9db690", conversation_id="4b28e333-5f5c-4a04-a242-ee9329edd51a", question="What's the capital of France?", answer="Paris", created_at=datetime.now(), updated_at=datetime.now(), response_time=500)
        # print(qa_handler.create_qa_pair(new_qa))

        # print(qa_handler.get_qa_pair_by_conversation_id("convo_456"))

        # print(qa_handler.get_k_most_recent_qa_pair_by_conversation_id("convo_456", 3))

        # updated_qa_pair = QAPair(question="Updated question", answer="Updated answer", response_time=300)
        # print(qa_handler.update_qa_pair_by_id(qa_pair_id="68b92dc2ad7283ee6ba67668", qa_pair=updated_qa_pair))

        # print(qa_handler.delete_qa_pair_by_id(qa_pair_id="68b90226e6f59bb7290ad890"))
