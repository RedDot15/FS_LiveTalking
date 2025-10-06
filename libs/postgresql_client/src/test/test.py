from __future__ import annotations
import os
from dotenv import load_dotenv
from postgresql_client import PostgreSQL
from logger import get_logger
from postgresql_client.settings import PostgresSettings

logger = get_logger(__name__)

load_dotenv()

POSTGRES__USER = os.getenv("POSTGRES__USER")
POSTGRES__PASSWORD = os.getenv("POSTGRES__PASSWORD")
POSTGRES__DB = os.getenv("POSTGRES__DB")
POSTGRES__HOST = os.getenv("POSTGRES__HOST")
POSTGRES__PORT = os.getenv("POSTGRES__PORT")

postgres = PostgreSQL(
    postgres_settings=PostgresSettings(
        user=POSTGRES__USER,
        password=POSTGRES__PASSWORD,
        host=POSTGRES__HOST,
        port=POSTGRES__PORT,
        db=POSTGRES__DB,
    )
)

# Từ đây xuống chưa fix
if __name__ == "__main__":

    # p = PermissionSchema(name='DELETE_USER')
    # target_permission_id = '01fbff89-e055-4b8d-a24b-09cf1e6c8bcb'

    # r = RoleSchema(name='ADMIN_2')
    # permission_ids = [target_permission_id]
    # target_role_id = 'ba44f815-48c3-49d7-8f40-8d189154b8ae'

    # u = UserSchema(username='admin2', password='admin', name='admin', avatar_url='url', email='sample@mail.com', phone_number='03632738')
    # role_ids = [target_role_id]
    # target_user_id = '1653d2f6-f1fa-4b66-83fe-08dde61d3bbe'

    with postgres.get_session() as session:
        # print(postgres.insert_permission(session, p))
        # print(postgres.get_permission(session))
        # print(postgres.get_permission_by_id(session, target_permission_id))
        # print(postgres.update_permission(session, p, target_permission_id))
        # print(postgres.delete_permission(session, target_permission_id))

        # print(postgres.insert_role(session, r, permission_ids))
        print(postgres.get_role(session))
        # print(postgres.get_role_by_id(session, target_role_id))
        # print(postgres.update_role(session, r, target_role_id, permission_ids))
        # print(postgres.delete_role(session, target_role_id))

        # print(postgres.insert_user(session, u, role_ids))
        # print(postgres.get_user(session))
        # print(postgres.get_user_by_id(session, target_user_id))
        # print(postgres.update_user(session, u, target_user_id, role_ids))
        # print(postgres.delete_user(session, target_user_id))
