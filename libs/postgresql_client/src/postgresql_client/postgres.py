from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from contextlib import contextmanager
from functools import cached_property

from .model import Base
from .controller import (
    UserController,
    RoleController,
    PermissionController
)

class PostgreSQL(
    UserController,
    RoleController,
    PermissionController,
):
    def __init__(self, user, password, host, db, port):
        self.user = user
        self.password = password
        self.host = host
        self.db = db
        self.port = port
        
    @cached_property
    def sessionmaker(self) -> sessionmaker:
        engine = create_engine(f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}')
        Base.metadata.create_all(engine)
        return sessionmaker(autoflush=False, bind=engine)

    @contextmanager
    def get_session(self):
        try:
            session: Session = self.sessionmaker()
            yield session
        finally:
            session.close()
