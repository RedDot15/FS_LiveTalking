from base import BaseModel

class MongoSettings(BaseModel):
    db: str
    username: str
    password: str
    host: str
    port: int
    