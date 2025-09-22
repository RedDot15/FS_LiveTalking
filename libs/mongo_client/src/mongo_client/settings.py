from base import CustomBaseModel

class MongoSettings(CustomBaseModel):
    db: str
    username: str
    password: str
    host: str
    port: int
    