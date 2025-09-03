from __future__ import annotations
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy import delete
from sqlalchemy.orm import Session
from collections.abc import Sequence

from structlog.stdlib import BoundLogger

from ..model import (
    Base,
    BaseSchema
)

def _get_data(logger: BoundLogger, 
         model_cls: type[Base],
         schema_cls: type[BaseSchema],
         session: Session,
         filter: dict[str, object] | None = None,
         order_by: Sequence | None = None,
         limit: int | None = None):
    try:
        statement = select(model_cls)
        if filter:
            statement = statement.filter_by(**filter)
        if order_by:
            statement = statement.order_by(*order_by)
        if limit:
            statement = statement.limit(limit)
        
        objs = session.scalars(statement=statement).all()
        
        if len(objs) == 0:
            return None
        return [schema_cls.model_validate(obj) for obj in objs]
    
    except Exception as e:
        logger.exception(
            f'Error fetching {schema_cls}: {e}',
            filter=filter,
            limit=limit,
        )
        raise e
    
def _get_data_by_id(logger: BoundLogger,
                    model_cls: type[Base],
                    schema_cls: type[BaseSchema],
                    session: Session,
                    id: str) -> BaseSchema:
    
    try:
        obj = session.get(model_cls, id)
        if obj:
            return schema_cls.model_validate(obj)
        else:
            logger.info(f'No {schema_cls} found with id: {id}')
            return None
    
    except Exception as e:
        logger.exception(
            f'Error fetching {schema_cls}: {e}',
        )
        raise e


def _insert(logger: BoundLogger, 
            model_cls: type[Base], 
            schema_cls: type[BaseSchema],
            session: Session,
            data: BaseSchema) -> BaseSchema:
    
    try:
        obj = model_cls(**data.model_dump(exclude_none=True))
        obj.id = uuid4()
        session.add(obj)
        session.commit()
        session.refresh(obj)
        
        return schema_cls.model_validate(obj)
    
    except Exception as e:
        logger.exception(f'Error inserting {schema_cls}: {e}', channel=data)
        raise e
    
def _update(logger: BoundLogger, 
            model_cls: type[Base],
            schema_cls: type[BaseSchema],
            session: Session,
            data: BaseSchema,
            id: str) -> BaseSchema:
    
    try:
        obj = session.get(model_cls, id)
        if obj:
            for key, value in data.model_dump(exclude_none=True, exclude={'id'}).items():
                if value is not None:
                    setattr(obj, key, value)
                    
            session.commit()
            session.refresh(obj)
            
            return schema_cls.model_validate(obj)
        
        else:
            logger.info(f'No {schema_cls} found with id: {data.id}')
            return None
        
    except Exception as e:
        logger.exception(f'Error updating {schema_cls}: {e}', channel=data)
        raise e
        
def _delete(logger: BoundLogger,
            model_cls: type[Base],
            schema_cls: type[BaseSchema],
            session: Session,
            id: str) -> BaseSchema:
    
    try:
        obj = session.get(model_cls, id)
        if obj:
            session.delete(obj)
            session.commit()

            return schema_cls.model_validate(obj)
        else:
            logger.info(f'No {schema_cls} found with id: {id}')
            return None
        
    except Exception as e:
        logger.exception(f'Error deleting {schema_cls}: {e}', id=id)
        raise e