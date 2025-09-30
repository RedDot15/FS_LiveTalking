from base import BaseModel

class MongoSettings(BaseModel):
    db: str
    user: str
    password: str
    host: str
    port: int
    