from __future__ import annotations

from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from shared.base import BaseModel
from shared.base import BaseService

from domain.generate_video import  GenerateVideoInput
from domain.generate_video import  GenerateVideoService

from shared.logger import get_logger

logger = get_logger(__name__)

class SadTalkerServiceInput(BaseModel):
    bucket_name: str
    character_name: str
    image_url: str
    audio_url: str

class SadTalkerServiceOutput(BaseModel):
    save_path: str

class SadTalkerService(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Any = Field(exclude=True)
    settings: Any = Field(exclude=True)

    @property
    def genetate_video_service(self) -> GenerateVideoService:
        return GenerateVideoService(
            settings=self.settings.sadtalker,
            minio_client=self.request.app.state.minio_client
        )


    def process(self, input: SadTalkerServiceInput) -> SadTalkerServiceOutput:
        
        try:
        
            result = self.genetate_video_service.process(
                input=GenerateVideoInput(
                    bucket_name=input.bucket_name,
                    character_name=input.character_name,
                    image_url=input.image_url,
                    audio_url=input.audio_url
                )
            )
            
            return SadTalkerServiceOutput(
                save_path=result.save_path
            )
            
        except Exception as e:
            logger.error(f"Error in SadTalkerService: {e}")
            raise e

        return result