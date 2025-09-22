from __future__ import annotations
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from contextlib import contextmanager
from functools import cached_property

from base import BaseService

from .model import CustomBaseModel
from .controller import UserController, RoleController, PermissionController
from .settings import PostgresSettings


class PostgreSQL(UserController, RoleController, PermissionController, BaseService):
    postgres_settings: PostgresSettings

    @cached_property
    def sessionmaker(self) -> sessionmaker:
        engine = create_engine(
            f"postgresql+psycopg2://{self.postgres_settings.user}:{self.postgres_settings.password}@{self.postgres_settings.host}:{self.postgres_settings.port}/{self.postgres_settings.db}"
        )
        CustomBaseModel.metadata.create_all(engine)
        return sessionmaker(autoflush=False, bind=engine)

    @contextmanager
    def get_session(self):
        try:
            session: Session = self.sessionmaker()
            yield session
        finally:
            session.close()

    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used in MongoDBHandler.")
