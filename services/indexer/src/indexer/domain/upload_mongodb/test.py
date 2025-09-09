from .service import CharacterMongoDBInputs, CharacterMongoDBOutputs, CharacterUploadMongoDBService
from mongo_client import MongoSettings, MongoDBHandler
from mongo_client.model import Character
from dotenv import load_dotenv
load_dotenv()
import os

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

tao = CharacterUploadMongoDBService(
    db_handler=db_handler
)

inputs = CharacterMongoDBInputs(
    name = "Minh",
    avatar_url="URL"
)
response = tao.upload_to_mongo(
    inputs=inputs
)
print(response)