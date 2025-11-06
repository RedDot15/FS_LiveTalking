from __future__ import annotations

from base import BaseModel
from base import BaseService
import os

class OsMakedirsInput(BaseModel):
    
    path_list: list
    

class OsMakedirsService(BaseService):
    
    def process(self, input: OsMakedirsInput) -> None:
        for path in input.path_list:
            os.makedirs(path, exist_ok=True)