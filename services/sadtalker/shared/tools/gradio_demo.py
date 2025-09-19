import torch, uuid
import os, sys, shutil
from shared.logger import get_logger
from shared.tools.features.preprocess import CropAndExtract
from shared.tools.test_audio2coeff import Audio2Coeff
from shared.tools.facerender.animate import AnimateFromCoeff
from shared.tools.generate_batch import get_data
from shared.tools.generate_facerender_batch import get_facerender_data

from shared.tools.features.init_path import init_path

from pydub import AudioSegment

logger = get_logger(__name__)


def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    logger.info(f"Converting MP3 to WAV: {mp3_filename} -> {wav_filename} with frame_rate={frame_rate}")
    try:
        mp3_file = AudioSegment.from_file(file=mp3_filename)
        mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")
        logger.info(f"Successfully converted MP3 to WAV: {wav_filename}")
    except Exception as e:
        logger.error(f"Failed to convert MP3 to WAV: {e}")
        raise


class SadTalker():

    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):
        logger.info(f"Initializing SadTalker with checkpoint_path='{checkpoint_path}', config_path='{config_path}', lazy_load={lazy_load}")

        if torch.cuda.is_available() :
            device = "cuda"
            logger.info("CUDA is available, using GPU")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
        
        self.device = device

        os.environ['TORCH_HOME']= checkpoint_path
        logger.debug(f"Set TORCH_HOME environment variable to: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        logger.info("SadTalker initialization completed")
      

    def test(self, source_image, driven_audio, preprocess='crop', 
        still_mode=False,  use_enhancer=False, batch_size=1, size=256, 
        pose_style = 0, exp_scale=1.0, 
        use_ref_video = False,
        ref_video = None,
        ref_info = None,
        use_idle_mode = False,
        length_of_audio = 0, use_blink=True,
        result_dir='./results/'):

        logger.info(f"Starting SadTalker test with source_image='{source_image}', driven_audio='{driven_audio}', preprocess='{preprocess}', still_mode={still_mode}, use_enhancer={use_enhancer}, batch_size={batch_size}, size={size}")
        logger.debug(f"Additional parameters: pose_style={pose_style}, exp_scale={exp_scale}, use_ref_video={use_ref_video}, use_idle_mode={use_idle_mode}, length_of_audio={length_of_audio}, use_blink={use_blink}")

        self.sadtalker_paths = init_path(self.checkpoint_path, self.config_path, size, False, preprocess)
        logger.debug(f"Initialized sadtalker paths: {self.sadtalker_paths}")
        print(self.sadtalker_paths)
        logger.info("Initializing models for SadTalker processing")
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)
        logger.info("Models initialized successfully")

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Created save directory: {save_dir}")

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)
        logger.debug(f"Created input directory: {input_dir}")

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.move(source_image, input_dir)
        logger.info(f"Moved source image to: {pic_path}")

        if driven_audio is not None and os.path.isfile(driven_audio):
            logger.info(f"Processing driven audio: {driven_audio}")
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  

            #### mp3 to wav
            if '.mp3' in audio_path:
                logger.info("Converting MP3 to WAV format")
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
                logger.info(f"Converted audio path: {audio_path}")
            else:
                shutil.move(driven_audio, input_dir)
                logger.info(f"Moved audio file to: {audio_path}")

        elif use_idle_mode:
            logger.info(f"Using idle mode with audio length: {length_of_audio} seconds")
            audio_path = os.path.join(input_dir, 'idlemode_'+str(length_of_audio)+'.wav') ## generate audio from this new audio_path
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)  #duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
            logger.info(f"Generated silent audio file: {audio_path}")
        else:
            logger.warning(f"No driven audio provided, using ref video mode: use_ref_video={use_ref_video}, ref_info={ref_info}")
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == 'all'

        if use_ref_video and ref_info == 'all': # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname+'.wav')
            print('new audiopath:',audio_path)
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s"%(ref_video, audio_path)
            os.system(cmd)        

        os.makedirs(save_dir, exist_ok=True)
        
        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir, preprocess, True, size)
        
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print('using ref video for genreation')
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_video_coeff_path, _, _ =  self.preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == 'pose':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':            
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise('error in refinfo')
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        #audio2ceoff
        if use_ref_video and ref_info == 'all':
            coeff_path = ref_video_coeff_path # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=still_mode, idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink) # longer audio?
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, preprocess=preprocess, size=size, expression_scale = exp_scale)
        return_path = self.animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size)
        video_name = data['video_name']
        print(f'The generated video is named {video_name} in {save_dir}')
        logger.info(f'Successfully generated video: {video_name} in directory: {save_dir}')

        logger.info("Cleaning up models and freeing memory")
        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache and synchronized")
            
        import gc; gc.collect()
        logger.debug("Performed garbage collection")
        
        logger.info(f"SadTalker processing completed successfully. Return path: {return_path}")
        return return_path

    