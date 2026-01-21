"""
FastAPI backend server for Urdu Speech Translation UI.
Provides REST API endpoints for the frontend.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import shutil
import wave
import numpy as np
from scipy import signal

# Import project modules
from speech_to_text import transcribe_urdu_audio
from speech_to_text_multi import transcribe_audio_multi
from translate_text import translate_urdu_to_target
from translate_text_multi import translate_text_multi
from text_to_speech import text_to_speech_piper
from text_cleaner import clean_for_translation, is_valid_transcription
from language_codes import CODE_TO_LANGUAGE_NAME

app = FastAPI(title="Urdu Speech Translation API", version="1.0.0")

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class TranslationRequest(BaseModel):
    urdu_text: str
    target_language: str = "English"


class ProcessRequest(BaseModel):
    target_language: str = "English"


class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Urdu Speech Translation API", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


# Get supported languages
@app.get("/api/languages")
async def get_languages():
    """Get list of all supported target languages."""
    languages = [
        {"code": code, "name": name}
        for code, name in sorted(CODE_TO_LANGUAGE_NAME.items(), key=lambda x: x[1])
    ]
    return {"languages": languages}


# Upload and process audio file
@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    target_language: str = "English"
):
    """
    Process uploaded audio: Transcribe → Translate → TTS.
    
    Returns:
        - urdu_text: Transcribed Urdu text
        - translated_text: Translated text
        - audio_url: URL to download the generated audio
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_audio_path = tmp_file.name
        
        # Step 1: Transcribe
        urdu_text = transcribe_urdu_audio(temp_audio_path)
        
        if not urdu_text:
            raise HTTPException(status_code=400, detail="Transcription failed")
        
        # Clean transcription
        cleaned_urdu = clean_for_translation(urdu_text)
        
        if not is_valid_transcription(cleaned_urdu):
            raise HTTPException(
                status_code=400,
                detail="Transcription appears to be noise or invalid"
            )
        
        # Step 2: Translate
        translated_text = translate_urdu_to_target(cleaned_urdu, target_language)
        
        if not translated_text:
            raise HTTPException(status_code=400, detail="Translation failed")
        
        # Step 3: Text-to-Speech
        output_audio_path = os.path.join(tempfile.gettempdir(), f"output_{os.urandom(8).hex()}.wav")
        audio_file = text_to_speech_piper(translated_text, target_language, output_audio_path)
        
        if not audio_file or not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Text-to-speech failed")
        
        # Clean up temp input file
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        return {
            "status": "success",
            "urdu_text": cleaned_urdu,
            "translated_text": translated_text,
            "audio_url": f"/api/audio/{os.path.basename(audio_file)}",
            "audio_file": audio_file
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# Upload audio for transcription only
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to Urdu text only."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_audio_path = tmp_file.name
        
        urdu_text = transcribe_urdu_audio(temp_audio_path)
        
        if not urdu_text:
            raise HTTPException(status_code=400, detail="Transcription failed")
        
        cleaned_urdu = clean_for_translation(urdu_text)
        
        # Clean up
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        return {
            "status": "success",
            "urdu_text": cleaned_urdu
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


# Translate text
@app.post("/api/translate")
async def translate_text(request: TranslationRequest):
    """Translate Urdu text to target language."""
    try:
        translated_text = translate_urdu_to_target(request.urdu_text, request.target_language)
        
        if not translated_text:
            raise HTTPException(status_code=400, detail="Translation failed")
        
        return {
            "status": "success",
            "translated_text": translated_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


# Text-to-Speech
@app.post("/api/tts")
async def text_to_speech(request: TranslationRequest):
    """Convert text to speech."""
    try:
        output_audio_path = os.path.join(
            tempfile.gettempdir(),
            f"tts_{os.urandom(8).hex()}.wav"
        )
        
        audio_file = text_to_speech_piper(
            request.translated_text if hasattr(request, 'translated_text') else request.urdu_text,
            request.target_language,
            output_audio_path
        )
        
        if not audio_file or not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Text-to-speech failed")
        
        return {
            "status": "success",
            "audio_url": f"/api/audio/{os.path.basename(audio_file)}",
            "audio_file": audio_file
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


# Serve audio files
@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files."""
    audio_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=filename
    )


# Process audio from base64 (for browser recording)
class ProcessBase64Request(BaseModel):
    audio_data: str
    target_language: str = "English"

# Conversation mode request (bidirectional translation)
class ConversationRequest(BaseModel):
    audio_data: str
    source_language: str  # Language being spoken (e.g., "Urdu" or "French")
    target_language: str  # Language to translate to (e.g., "French" or "Urdu")
    conversation_mode: bool = True  # Enable conversation mode

@app.post("/api/conversation")
async def process_conversation(request: ConversationRequest):
    """
    Process audio in conversation mode (bidirectional translation).
    
    This endpoint handles both directions:
    - Urdu → Target Language (e.g., Urdu → French)
    - Target Language → Urdu (e.g., French → Urdu)
    
    Args:
        audio_data: Base64-encoded audio data
        source_language: Language being spoken
        target_language: Language to translate to
        conversation_mode: Enable conversation features
    
    Returns:
        - source_text: Transcribed text in source language
        - translated_text: Translated text in target language
        - audio_base64: Base64-encoded audio of translated speech
    """
    try:
        import base64
        
        audio_data = request.audio_data
        source_language = request.source_language
        target_language = request.target_language
        
        # Extract base64 data
        if "," in audio_data:
            base64_data = audio_data.split(",")[1]
        else:
            base64_data = audio_data
        
        # Decode base64
        audio_bytes = base64.b64decode(base64_data)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_audio_path = tmp_file.name
        
        # Step 1: Transcribe audio in source language
        if source_language.lower() in ["urdu", "ur"]:
            source_text = transcribe_urdu_audio(temp_audio_path)
        else:
            source_text = transcribe_audio_multi(temp_audio_path, source_language)
        
        if not source_text:
            raise HTTPException(status_code=400, detail="Transcription failed")
        
        # Clean transcription
        if source_language.lower() in ["urdu", "ur"]:
            cleaned_source = clean_for_translation(source_text)
            if not is_valid_transcription(cleaned_source):
                raise HTTPException(
                    status_code=400,
                    detail="Transcription appears to be noise or invalid"
                )
        else:
            cleaned_source = source_text.strip()
        
        # Step 2: Translate from source to target language
        translated_text = translate_text_multi(
            cleaned_source,
            source_language,
            target_language
        )
        
        if not translated_text:
            raise HTTPException(status_code=400, detail="Translation failed")
        
        # Step 3: Text-to-Speech in target language
        output_audio_path = os.path.join(
            tempfile.gettempdir(),
            f"output_{os.urandom(8).hex()}.wav"
        )
        audio_file = text_to_speech_piper(translated_text, target_language, output_audio_path)
        
        if not audio_file or not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Text-to-speech failed")
        
        # Read audio file and convert to base64 for response
        with open(audio_file, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        try:
            os.unlink(temp_audio_path)
            os.unlink(audio_file)
        except:
            pass
        
        return {
            "status": "success",
            "source_text": cleaned_source,
            "source_language": source_language,
            "translated_text": translated_text,
            "target_language": target_language,
            "audio_base64": f"data:audio/wav;base64,{audio_base64}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/process-base64")
async def process_base64_audio(request: ProcessBase64Request):
    """
    Process base64-encoded audio data.
    Audio should be in format: data:audio/wav;base64,<base64_data>
    """
    try:
        import base64
        
        audio_data = request.audio_data
        target_language = request.target_language
        
        # Extract base64 data
        if "," in audio_data:
            base64_data = audio_data.split(",")[1]
        else:
            base64_data = audio_data
        
        # Decode base64
        audio_bytes = base64.b64decode(base64_data)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_audio_path = tmp_file.name
        
        # Process audio (same as /api/process)
        urdu_text = transcribe_urdu_audio(temp_audio_path)
        
        if not urdu_text:
            raise HTTPException(status_code=400, detail="Transcription failed")
        
        cleaned_urdu = clean_for_translation(urdu_text)
        
        if not is_valid_transcription(cleaned_urdu):
            raise HTTPException(
                status_code=400,
                detail="Transcription appears to be noise or invalid"
            )
        
        translated_text = translate_urdu_to_target(cleaned_urdu, target_language)
        
        if not translated_text:
            raise HTTPException(status_code=400, detail="Translation failed")
        
        output_audio_path = os.path.join(
            tempfile.gettempdir(),
            f"output_{os.urandom(8).hex()}.wav"
        )
        audio_file = text_to_speech_piper(translated_text, target_language, output_audio_path)
        
        if not audio_file or not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Text-to-speech failed")
        
        # Read audio file and convert to base64 for response
        with open(audio_file, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        try:
            os.unlink(temp_audio_path)
            os.unlink(audio_file)
        except:
            pass
        
        return {
            "status": "success",
            "urdu_text": cleaned_urdu,
            "translated_text": translated_text,
            "audio_base64": f"data:audio/wav;base64,{audio_base64}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
