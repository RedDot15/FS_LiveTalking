from __future__ import annotations

import os
import shutil
from time import  strftime
import requests
import tempfile

from shared.settings import SadTalkerSettings
from shared.tools import init_path
from shared.tools import get_data
from shared.tools import get_facerender_data
from shared.tools import CropAndExtract
from shared.tools import Audio2Coeff
from shared.tools import AnimateFromCoeff
from shared.base import BaseModel
from shared.base import BaseService
from shared.logger import get_logger

from infra.minio_client import MinioConnection

logger = get_logger(__name__)

class GenerateVideoInput(BaseModel):
    character_name: str
    image_url: str
    audio_url: str

class GenerateVideoOutput(BaseModel):
    save_path: str

class GenerateVideoService(BaseService):

    bucket_name: str
    settings: SadTalkerSettings
    minio_client: MinioConnection

    def process(self, input: GenerateVideoInput) -> GenerateVideoOutput:
        
        logger.info('Initializing paths and models for video generation')
        sadtalker_paths = init_path(
            checkpoint_dir=self.settings.checkpoint_dir, 
            config_dir=self.settings.config_dir, 
            size=self.settings.size, 
            preprocess=self.settings.preprocess
        )

        # Up to Minio
        save_dir = os.path.join(self.settings.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))

        logger.info("INITIATING CROP AND EXTRACT PROCESS MODEL")
        preprocess_model = CropAndExtract(sadtalker_paths, self.settings.device)
        
        logger.info("INITIATING AUDIO TO COEFFICIENTS PROCESS MODEL")
        audio_to_coeff = Audio2Coeff(sadtalker_paths,  self.settings.device)

        logger.info("INITIATING ANIMATE FROM COEFFICIENTS PROCESS MODEL")
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, self.settings.device)
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')

        os.makedirs(first_frame_dir, exist_ok=True)
        logger.info("3DMM Extraction for source image")

        image_path, audio_path = self.get_url_minio(
            bucket_name=self.bucket_name,
            character_name=input.character_name,
            image_url=input.image_url,
            audio_url=input.audio_url
        )

        logger.info("GENERATING DATA FROM MINIO URLS", extra={'image_path': image_path, 'audio_path': audio_path})
        
        # This one is for source image
        first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(
            image_path, 
            first_frame_dir, 
            self.settings.preprocess,
            source_image_flag=True, 
            pic_size=self.settings.size
        )

        if first_coeff_path is None:
            raise ValueError("Failed to extract 3DMM coefficients from the source image.")
        

        if self.settings.ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(self.settings.ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            
            logger.info('3DMM Extraction for the reference video providing eye blinking')

            ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(
                self.settings.ref_eyeblink, 
                ref_eyeblink_frame_dir, 
                self.settings.preprocess, 
                source_image_flag=False
            )
        else:
            ref_eyeblink_coeff_path=None

        if self.settings.ref_pose is not None:
            if self.settings.ref_pose == self.settings.ref_eyeblink: 
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(self.settings.ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                logger.info('3DMM Extraction for the reference video providing pose')
                
                ref_pose_coeff_path, _, _ =  preprocess_model.generate(
                    self.settings.ref_pose, 
                    ref_pose_frame_dir, 
                    self.settings.preprocess, 
                    source_image_flag=False
                )
        else:
            ref_pose_coeff_path=None
            

        #audio2coeff
        #This one is for audio driving
        logger.info('STARTING AUDIO TO COEFFICIENTS GENERATION')
        batch = get_data(
            first_coeff_path, 
            audio_path, 
            self.settings.device, 
            ref_eyeblink_coeff_path, 
            still=self.settings.still,
            idlemode=self.settings.idlemode)
        coeff_path = audio_to_coeff.generate(batch, save_dir, self.settings.pose_style, ref_pose_coeff_path)

        #coeff2video
        # This one is for audio
        logger.info('STARTING DATA PREPARATION FOR VIDEO GENERATION')
        data = get_facerender_data(
            coeff_path, 
            crop_pic_path, 
            first_coeff_path, 
            audio_path, 
            self.settings.batch_size, 
            self.settings.input_yaw, 
            self.settings.input_pitch, 
            self.settings.input_roll,
            expression_scale=self.settings.expression_scale, 
            still_mode=self.settings.still, 
            preprocess=self.settings.preprocess, 
            size=self.settings.size
        )
        
        # This one using source image
        logger.info('STARTING VIDEO GENERATION')
        result = animate_from_coeff.generate(
            data, 
            save_dir, 
            image_path, 
            crop_info,
            enhancer=self.settings.enhancer, 
            background_enhancer=self.settings.background_enhancer, 
            preprocess=self.settings.preprocess, 
            img_size=self.settings.size
        )
        
        logger.info('VIDEO GENERATION COMPLETED, SAVING AND UPLOADING TO MINIO')
        
        shutil.move(result, save_dir + '.mp4')
        video_path = save_dir + '.mp4'
        video_filename = os.path.basename(video_path)
        
        
        save_path = self.minio_client.put_object(
            bucket_name=self.bucket_name,
            src_file=video_path,
            des_folder_name=f'{input.character_name}/videos', 
            des_file_name=video_filename
        )
        
        logger.info('The generated video is saved at', extra={'video_path': video_path + '.mp4'})

        if not self.settings.verbose:
            shutil.rmtree(save_dir)
            
        save_path = "/".join(save_path.split("/")[2:])
            
        return GenerateVideoOutput(save_path=save_path)

    def get_url_minio(self, bucket_name: str, character_name: str, image_url: str, audio_url: str) -> list[str]:
        
        logger.info(
            'STARTING TO GENERATE PRESIGNED URLS FROM MINIO',
            extra={'bucket_name': bucket_name, 'image_url': image_url, 'audio_url': audio_url}
        )
        
        image_url_minio = character_name + '/' + image_url
        audio_url_minio = character_name + '/' + audio_url

        presigned_image_url = self.minio_client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=image_url_minio 
        )
        presigned_audio_url = self.minio_client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=audio_url_minio
        )

        logger.info('DONE GENERATING PRESIGNED URLS')

        def download_to_temp(url, original_filename):
            
            suffix = os.path.splitext(original_filename)[1] or ''
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

        image_local_path = download_to_temp(presigned_image_url, image_url_minio)
        audio_local_path = download_to_temp(presigned_audio_url, audio_url_minio)

        return [image_local_path, audio_local_path]