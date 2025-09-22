from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from .base_model import CustomBaseModel as CustomBaseModel

class BaseService(ABC, CustomBaseModel):
    @abstractmethod
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError()

class AsyncBaseService(ABC, CustomBaseModel):
    @abstractmethod
    async def process(self, inputs: Any) -> Any:
        raise NotImplementedError()
