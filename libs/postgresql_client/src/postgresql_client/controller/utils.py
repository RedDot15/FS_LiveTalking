from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from collections.abc import Sequence

from structlog.stdlib import BoundLogger

from ..model import CustomBaseModel


def _get_data(
    logger: BoundLogger,
    model_cls: type[CustomBaseModel],
    session: Session,
    filter: dict[str, object] | None = None,
    order_by: Sequence | None = None,
    limit: int | None = None,
):
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
        return [obj for obj in objs]

    except Exception as e:
        logger.exception(
            f"Error fetching {model_cls}: {e}",
            filter=filter,
            limit=limit,
        )
        raise e


def _get_data_by_id(
    logger: BoundLogger, model_cls: type[CustomBaseModel], session: Session, id: str
) -> CustomBaseModel:
    try:
        obj = session.get(model_cls, id)
        if obj:
            return obj
        else:
            logger.info(f"No {model_cls} found with id: {id}")
            return None

    except Exception as e:
        logger.exception(
            f"Error fetching {model_cls}: {e}",
        )
        raise e


def _insert(
    logger: BoundLogger, model_cls: type[CustomBaseModel], session: Session, data: CustomBaseModel
) -> CustomBaseModel:
    try:
        obj = data
        session.add(obj)
        session.commit()
        session.refresh(obj)

        return obj

    except Exception as e:
        logger.exception(f"Error inserting {model_cls}: {e}", channel=data)
        raise e


def _update(
    logger: BoundLogger, model_cls: type[CustomBaseModel], session: Session, data: CustomBaseModel
) -> CustomBaseModel:
    try:
        obj = session.get(model_cls, data.id)
        if obj:
            for key, value in data.model_dump(exclude_none=True).items():
                if value is not None:
                    setattr(obj, key, value)

            session.commit()
            session.refresh(obj)

            return obj

        else:
            logger.info(f"No {model_cls} found with id: {data.id}")
            return None

    except Exception as e:
        logger.exception(f"Error updating {model_cls}: {e}", channel=data)
        raise e


def _delete(
    logger: BoundLogger, model_cls: type[CustomBaseModel], session: Session, id: str
) -> CustomBaseModel:
    try:
        obj = session.get(model_cls, id)
        if obj:
            session.delete(obj)
            session.commit()

            return obj
        else:
            logger.info(f"No {model_cls} found with id: {id}")
            return None

    except Exception as e:
        logger.exception(f"Error deleting {model_cls}: {e}", id=id)
        raise e
