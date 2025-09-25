from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from shared.base import BaseModel
from shared.logger import get_logger
from application.tts_service import TTSService
from application.tts_service import TTSServiceInput
from application.tts_service import SynthesisFileRequest

logger = get_logger(__name__)
voice_clone_router = APIRouter(prefix='/v1')


@voice_clone_router.post('/voice_clone')
async def tts_stream(request: Request, tts_service_input: TTSServiceInput):
    """Stream TTS audio generation"""
    try:
        tts_service = TTSService(
            settings=request.app.state.settings,
            request=request
        )
        
    except Exception as e:
        logger.error(f"Error initializing TTS service: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def generator():
        async for chunk in tts_service.process(
            input=TTSServiceInput(
                text=tts_service_input.text,
                character_name=tts_service_input.character_name,
                language=tts_service_input.language,
                stream=True
            )
        ):
            # Check if client disconnected
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(generator(), media_type='audio/x-wav')
        
@voice_clone_router.post("/tts_to_file")
async def tts_to_file(request: Request, body: SynthesisFileRequest):
    """Generate TTS audio and save to file"""
    try:
        tts_service = TTSService(
            tts_wrapper=request.app.state.tts_wrapper
        )
        
    except Exception as e:
        logger.error(f"Error initializing TTS service: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    
    try:
    
        response = tts_service.process_text_to_speech(
            input=TTSServiceInput(
                text=body.text,
                speaker_wav=body.speaker_wav,
                language=body.language,
                file_name_or_path=body.file_name_or_path,
                stream=False
            )
        )
    except Exception as e:
        logger.error(f"Error during TTS file generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
            
    return {
        "message": "The audio was successfully made and stored.",
        "output_path": response.output_path,
        "processing_time": response.processing_time
    }

@voice_clone_router.get("/languages")
async def get_languages(request: Request):
    """Get supported languages"""
    try:
        tts_service = TTSService(
            tts_wrapper=request.app.state.tts_wrapper
        )

        languages = tts_service.get_supported_languages()
        return {"languages": languages}
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@voice_clone_router.get("/speakers")
async def get_speakers(request: Request):
    """Get available speakers"""
    try:
        tts_service = TTSService(
            tts_wrapper=request.app.state.tts_wrapper
        )

        speakers = tts_service.get_available_speakers()
        return {"speakers": speakers}
    except Exception as e:
        logger.error(f"Error getting speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")