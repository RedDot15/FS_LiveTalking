from __future__ import annotations

import os
import time
import torch
import torchaudio
import numpy as np
from datetime import datetime

from shared.base import BaseModel
from shared.base import BaseService

from shared.logger import get_logger

logger = get_logger(__name__)

class TTSWrapperInput(BaseModel):
    text: str
    speaker_name_or_path: str
    
    language: str
    file_name_or_path: str
    stream: bool = False
    speaker_folder: str = './speakers'
    output_folder: str = './output'
    model_source: str = 'local'  # 'local' or 'api'
    enable_cache_results: bool = True
    
    
class TTSWrapperOutput(BaseModel):
    output_file: str

class TTSWrapperService(BaseService):

    def process(self, input: TTSWrapperInput) -> TTSWrapperOutput:
        try:
            
            # it's a speaker name
            full_path = os.path.join(
                input.speaker_folder, 
                input.speaker_name_or_path
            )
            
            speaker_wav = self._get_speaker_wav(
                speaker_name_or_path=input.speaker_name_or_path,
                full_path=full_path
            )
            
            # Determine output path based on whether a full path or a file name was provided
            if os.path.isabs(input.file_name_or_path):
                # An absolute path was provided by user; use as is.
                output_file = input.file_name_or_path
            else:
                # Only a filename was provided; prepend with output folder.
                output_file = os.path.join(input.output_folder, input.file_name_or_path)

            # Check if 'text' is a valid path to a '.txt' file.
            if os.path.isfile(input.text) and input.text.lower().endswith('.txt'):
                with open(input.text, 'r', encoding='utf-8') as f:
                    input.text = f.read()

            # Generate unic name for cached result
            if input.enable_cache_results:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                file_name_or_path = timestamp + "_cache_" + file_name_or_path
                output_file = os.path.join(input.output_folder, file_name_or_path)

            # Replace double quotes with single, asterisks, carriage returns, and line feeds
            clear_text = self.clean_text(input.text)

            # Generate a dictionary of the parameters to use for caching.
            text_params = {
              'text': clear_text,
              'speaker_name_or_path': input.speaker_name_or_path,
              'language': input.language
            }

            # Check if results are already cached.
            cached_result = self.check_cache(text_params)

            if cached_result is not None:
                logger.info("Using cached result.")
                return cached_result  # Return the path to the cached result.

            self.switch_model_device() # Load to CUDA if lowram ON

            # Define generation if model via api or locally
            if input.model_source == "local":
                if input.stream:
                    async def stream_fn():
                        async for chunk in self.stream_generation(clear_text, input.speaker_name_or_path, speaker_wav, input.language, output_file):
                            yield chunk
                        self.switch_model_device()
                        # After generation completes successfully...
                        self.update_cache(text_params,output_file)
                    return stream_fn()
                else:
                    self.local_generation(clear_text, input.speaker_name_or_path, speaker_wav, input.language, output_file)
            else:
                self.api_generation(clear_text, speaker_wav, input.language, output_file)

            self.switch_model_device() # Unload to CPU if lowram ON

            # After generation completes successfully...
            self.update_cache(text_params,output_file)
            return output_file

        except Exception as e:
            raise e

    async def stream_generation(self, text: str, speaker_name: str, speaker_wav: str, language: str, output_file: str):
        # Log time
        generate_start_time = time.time()  # Record the start time of loading the model

        gpt_cond_latent, speaker_embedding = self.get_or_create_latents(speaker_name, speaker_wav)
        file_chunks = []

        chunks = self.model.inference_stream(
            text,
            language,
            speaker_embedding=speaker_embedding,
            gpt_cond_latent=gpt_cond_latent,
            **self.tts_settings, # Expands the object with the settings and applies them for generation
            stream_chunk_size=self.stream_chunk_size,
        )
        
        for chunk in chunks:
            if isinstance(chunk, list):
                chunk = torch.cat(chunk, dim=0)
            file_chunks.append(chunk)
            chunk = chunk.cpu().numpy()
            chunk = chunk[None, : int(chunk.shape[0])]
            chunk = np.clip(chunk, -1, 1)
            chunk = (chunk * 32767).astype(np.int16)
            yield chunk.tobytes()

        if len(file_chunks) > 0:
            wav = torch.cat(file_chunks, dim=0)
            torchaudio.save(output_file, wav.cpu().squeeze().unsqueeze(0), 24000)
        else:
            logger.warning("No audio generated.")

        generate_end_time = time.time()  # Record the time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time

        logger.info(f"Processing time: {generate_elapsed_time:.2f} seconds.")

    def get_or_create_latents(self, latents_cache: dict, speaker_name: str, speaker_wav: str):
        if speaker_name not in latents_cache:
            logger.info(f"creating latents for {speaker_name}: {speaker_wav}")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(speaker_wav)
            latents_cache[speaker_name] = (gpt_cond_latent, speaker_embedding)
        return latents_cache[speaker_name]

    def _get_wav_files(self, directory: str) -> list[str]:
        """ Finds all the wav files in a directory. """
        wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        return wav_files

    def _get_speaker_wav(self, speaker_name_or_path: str, full_path: str) -> str:

        """ Gets the speaker_wav(s) for a given speaker name. """
        
        if speaker_name_or_path.endswith('.wav'):
            # it's a file name
            if os.path.isabs(speaker_name_or_path):
                # absolute path; nothing to do
                speaker_wav = speaker_name_or_path
            else:
                # make it a full path
                speaker_wav = full_path
                
        else:
            
            wav_file = f"{full_path}.wav"
            if os.path.isdir(full_path):
                # multi-sample speaker
                speaker_wav = [ os.path.join(full_path,wav) for wav in full_path]
                if len(speaker_wav) == 0:
                    raise ValueError(f"no wav files found in {full_path}")
            elif os.path.isfile(wav_file):
                speaker_wav = wav_file
            else:
                raise ValueError(f"Speaker {speaker_name_or_path} not found.")

        return speaker_wav