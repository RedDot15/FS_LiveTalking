from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import tempfile
import requests

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field

from wav2lip.domain.osmakedirs import OsMakedirsInput
from wav2lip.domain.osmakedirs import OsMakedirsService

from wav2lip.domain.process_video import ProcessVideoInput
from wav2lip.domain.process_video import ProcessVideoService

from wav2lip.domain.face_detection import FaceDetectionInput
from wav2lip.domain.face_detection import FaceDetectionService

from wav2lip.domain.save_coords import SaveCoordsInput
from wav2lip.domain.save_coords import SaveCoordsService

logger = get_logger(__name__)

class Wav2lipApplicationInput(BaseModel):
    
    character_name: str
    video_url: str
    
class Wav2lipApplicationOutput(BaseModel):
    
    wav2lip_result_path: str
    
class Wav2lipApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    @property
    def osmakedirs_service(self) -> OsMakedirsService:
        return OsMakedirsService()
    
    @property
    def process_video_service(self) -> ProcessVideoService:
        return ProcessVideoService()

    @property
    def face_detection_service(self) -> FaceDetectionService:
        return FaceDetectionService()
    
    @property
    def save_coords_service(self) -> SaveCoordsService:
        return SaveCoordsService()

    def process(self, input: Wav2lipApplicationInput) -> Wav2lipApplicationOutput:
        
        video_path = self.download_video_from_minio(
            character_name=input.character_name,
            bucket_name=self.settings.bucket_name,
            video_url=input.video_url
        )
        
        base_dir = Path("/home/reunion/app/wav2lip_data")
        avatar_path = base_dir / "avatars" / input.character_name
        full_imgs_path = avatar_path / "full_imgs"
        face_imgs_path = avatar_path / "face_imgs"
        coords_path = avatar_path / "coords.pkl"
        
        try:

            self.osmakedirs_service.process(
                input=OsMakedirsInput(
                    path_list=[
                        avatar_path, 
                        full_imgs_path, 
                        face_imgs_path
                    ]
                )
            )
        except Exception as e:
            logger.error("Error creating directories", extra={e})
            raise e
        
        try:
        
            process_video_results = self.process_video_service.process(
                input=ProcessVideoInput(
                    video_path=video_path,
                    save_path=str(full_imgs_path),
                    ext='.png',
                    cut_frame=10000000
                )
            )
        except Exception as e:
            logger.error("Error processing video", extra={e})
            raise e
        
        try:
        
            face_det_results = self.face_detection_service.process(
                input=FaceDetectionInput(
                    images=process_video_results.frames,
                    batch_size=self.settings.face_det_batch_size,
                    pads=self.settings.pads,
                    nosmooth=self.settings.nosmooth,
                )
            )
        except Exception as e:  
            logger.error("Error in face detection", extra={e})
            raise e
        
        try:

            self.save_coords_service.process(
                input=SaveCoordsInput(
                    img_size=self.settings.img_size,
                    coords_path=str(coords_path),
                    face_det_results=face_det_results,
                    face_imgs_path=str(face_imgs_path)
                )
            )
        except Exception as e:
            logger.error("Error saving coordinates", extra={e})
            raise e
        
        if self.request.app.state.minio_client.check_file_name_exists(
            bucket_name=self.settings.bucket_name,
            file_name=f'{input.character_name}/avatars/coords.pkl'
        ):
            
            logger.info('REMOVING OLD AVATAR DATA FROM MINIO')
            
            self.request.app.state.minio_client.remove_folder(
                bucket_name=self.settings.bucket_name,
                folder_name=f'{input.character_name}/avatars'
            )
        
        try:
        
            avatars_path = self.request.app.state.minio_client.put_folder(
                bucket_name=self.settings.bucket_name,
                des_folder_name=input.character_name + '/avatars',
                local_folder_path=avatar_path
            )
            
        except Exception as e:
            logger.error("Error uploading avatars to MinIO", extra={e})
            raise e

        return Wav2lipApplicationOutput(
            wav2lip_result_path=avatars_path
        )
        
    def download_video_from_minio(self, character_name: str, bucket_name: str, video_url: str, suffix: str = ".mp4") -> None:

        video_url = character_name + '/' + video_url

        url = self.request.app.state.minio_client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=video_url
        )
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file.flush()
            
            return temp_file.name
        finally:
            temp_file.close()