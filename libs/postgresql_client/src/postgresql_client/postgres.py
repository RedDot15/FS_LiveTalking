from __future__ import annotations
import os
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.base import Engine 

from contextlib import contextmanager
from functools import cached_property

from base import BaseService
from .controller import (
    UserController,
    RoleController,
    PermissionController
)
from .settings import PostgresSettings
from .model import Base

SQL_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # Get current directory of postgres.py
    "scripts",  # Navigate to the 'scripts' subdirectory
    "data.sql"  # Specify the SQL file name
)

class PostgreSQL(UserController, RoleController, PermissionController, BaseService):
    postgres_settings: PostgresSettings

    def _run_sql_script(self, engine: Engine):
        """Helper to run the data.sql script using the raw DBAPI connection."""
        try:
            with open(SQL_SCRIPT_PATH, 'r') as file:
                sql_script = file.read()
            
            with engine.connect() as connection:
                raw_dbapi_connection = connection.connection 
                cursor = raw_dbapi_connection.cursor()
                
                statements = [s.strip() for s in sql_script.split(';') if s.strip()]
                
                if not statements:
                    print("Warning: SQL script is empty or only whitespace.")
                    return

                for statement in statements:
                    # Execute each statement individually
                    cursor.execute(statement)

                # Commit the transaction to apply the changes
                raw_dbapi_connection.commit() 
                
                print(f"Successfully ran SQL script: {SQL_SCRIPT_PATH}")

        except FileNotFoundError:
            print(f"Warning: SQL script not found at {SQL_SCRIPT_PATH}")
        except Exception as e:
            print(f"Error running SQL script: {e}")
            raise 

    @cached_property
    def sessionmaker(self) -> sessionmaker:
        engine = create_engine(
            f"postgresql+psycopg2://{self.postgres_settings.user}:{self.postgres_settings.password}@{self.postgres_settings.host}:{self.postgres_settings.port}/{self.postgres_settings.db}"
        )

        # Create tables
        Base.metadata.create_all(engine)

        # Populate initial data
        self._run_sql_script(engine) 

        return sessionmaker(autoflush=False, bind=engine)

    @contextmanager
    def get_session(self):
        session = None
        try:
            session: Session = self.sessionmaker()
            yield session
        finally:
            if session:
                session.close()

    def process(self, inputs: Any) -> Any:
        raise NotImplementedError("This method is not used in MongoDBHandler.")