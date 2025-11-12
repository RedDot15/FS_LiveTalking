"""
TTS Application Service - Contains business logic for Text-to-Speech operations
"""
from __future__ import annotations

from typing import Optional, AsyncGenerator
import os
import glob
import time

from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from shared.base import BaseModel
from shared.base import BaseService
from shared.logger import get_logger
from infra.xtts.tts_funcs import supported_languages

logger = get_logger(__name__)

class SynthesisFileRequest(BaseModel):
    text: str
    speaker_wav: str 
    language: str
    file_name_or_path: str  

class SynthesisStreamRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str

class TTSServiceInput(BaseModel):
    text: str
    # speaker_wav: str
    language: str
    file_name_or_path: Optional[str] = "output.wav"
    stream: bool = False
    character_name: str


class TTSServiceOutput(BaseModel):
    output_path: str
    processing_time: float


class TTSService(BaseService):
    """
    TTS Application Service - handles business logic for text-to-speech operations
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Any = Field(exclude=True)
    settings: Any = Field(exclude=True)
    
    def validate_language(self, language: str) -> str:
        """
        Validate and normalize language code
        Maps Vietnamese to English for tokenizer compatibility
        """
        language = language.lower()
        
        # Map Vietnamese to English due to tokenizer limitations
        if language == 'vi':
            logger.info("Mapping Vietnamese language to English for tokenizer compatibility")
            language = 'en'
        
        if language not in supported_languages:
            raise ValueError(f"Language '{language}' is not supported. Supported languages: {list(supported_languages.keys())}")
        
        return language
    
    def validate_speaker(self, speaker_wav: str) -> str:
        """
        Validate speaker name - actual validation happens in TTSWrapper
        """
        if not speaker_wav or speaker_wav.strip() == "":
            raise ValueError("Speaker name cannot be empty")
        return speaker_wav
    
    def process_text_to_speech(self, input: TTSServiceInput) -> TTSServiceOutput:
        """
        Process text-to-speech conversion (non-streaming)
        """
        start_time = time.time()
        
        # Validate inputs
        language = self.validate_language(input.language)
        speaker_name = self.validate_speaker(input.speaker_wav)
        
        logger.info(f"Processing TTS: text_length={len(input.text)}, language='{language}', speaker='{input.speaker_wav}'")
        
        # Process TTS
        output_path = self.tts_wrapper.process_tts_to_file(
            text=input.text,
            speaker_name_or_path=speaker_name,
            language=language,
            file_name_or_path=input.file_name_or_path,
            stream=False
        )
        
        processing_time = time.time() - start_time
        logger.info(f"TTS processing completed in {processing_time:.2f} seconds")
        
        return TTSServiceOutput(
            output_path=output_path,
            processing_time=processing_time
        )
    
    async def process(self, input: TTSServiceInput) -> AsyncGenerator[bytes, None]:
        """
        Process text-to-speech conversion (streaming)
        """
        
        try:
            audio_folder = self.request.app.state.minio_client.get_folder(
                bucket_name=self.settings.bucket_name,
                des_folder_name=input.character_name,
                prefix=self.settings.prefix,
                local_folder_path=self.settings.local_folder_path
            )
            
            audio_folder = audio_folder + '/' + input.character_name + '/' + self.settings.prefix
            logger.info(f"Audio folder retrieved from MinIO: {audio_folder}")
            
        except Exception as e:
            logger.error(f"Error accessing MinIO folder: {e}")
            raise
        
        mp3_files = glob.glob(os.path.join(audio_folder, "*.mp3"))
        wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
        
        audio_files = mp3_files + wav_files
        
        if not audio_files:
            raise ValueError(f"No audio files found in folder: {audio_folder}")
        
        speaker_wav = os.path.abspath(audio_files[0])
        logger.info(f"Using speaker audio file: {speaker_wav}")
        
        # Validate inputs
        language = self.validate_language(input.language)
        speaker_name = self.validate_speaker(speaker_wav)

        logger.info(f"Processing TTS stream: text_length={len(input.text)}, language='{language}', speaker='{speaker_name}'")

        # Get WAV header first
        yield self.request.app.state.tts_wrapper.get_wav_header()
        
        # Process streaming TTS
        chunks = self.request.app.state.tts_wrapper.process_tts_to_file(
            text=input.text,
            speaker_name_or_path=speaker_name,
            language=language,
            file_name_or_path=input.file_name_or_path,
            stream=True
        )
        
        async for chunk in chunks:
            yield chunk
    
    def get_supported_languages(self) -> dict:
        """Get list of supported languages"""
        return supported_languages.copy()
    
    def get_available_speakers(self) -> list:
        """Get list of available speakers"""
        return self.request.app.state.tts_wrapper.get_speakers()