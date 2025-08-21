from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from gtts import gTTS
from agents.agent_graph import agent_graph
from langchain_core.messages import HumanMessage, AIMessage
import json
import speech_recognition as sr
import requests
import io
import os
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import re
from datetime import datetime
import base64
import subprocess
import sys
from pathlib import Path
import shutil

app = FastAPI(title="SecurAssist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = sr.Recognizer()

# Initialize TTS APIs in order of preference
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel's voice as default
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

SPEECHIFY_API_KEY = os.getenv("SPEECHIFY_API_KEY")
SPEECHIFY_VOICE_ID = os.getenv("SPEECHIFY_VOICE_ID", "default")
SPEECHIFY_BASE_URL = os.getenv("SPEECHIFY_BASE_URL", "https://api.speechify.com")

RESEMBLE_AI_API_KEY = os.getenv("RESEMBLE_AI_API_KEY")
RESEMBLE_AI_VOICE_UUID = os.getenv("RESEMBLE_AI_VOICE_UUID", "55592656")
RESEMBLE_AI_BASE_URL = "https://f.cluster.resemble.ai"

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

@app.on_event("startup")
async def startup_event():
    """Check for required dependencies on startup"""
    if not check_ffmpeg_installed():
        print("⚠️  WARNING: FFmpeg is not installed. Audio conversion may not work.")
        print("   To install FFmpeg on Render, add 'ffmpeg' to apt.txt")
    else:
        print("✅ FFmpeg is installed and available")
        
    # Check other dependencies
    try:
        import speech_recognition as sr
        import pydub
        print("✅ All audio dependencies loaded successfully")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")

def normalize_text_for_tts(text):
    """
    Clean text to be TTS-friendly by removing/replacing problematic characters
    """
    replacements = {
        '*': ' bullet point ',
        '~': ' approximately ',
        '_': ' ',
        '\\': ' or ',
        '/': ' or ',
        '|': ' or ',
        '#': ' number ',
        '@': ' at ',
        '%': ' percent ',
        '^': ' to the power of ',
        '&': ' and ',
        '+': ' plus ',
        '=': ' equals ',
        '<': ' less than ',
        '>': ' greater than ',
        '[': ' ',
        ']': ' ',
        '{': ' ',
        '}': ' ',
        '`': ' ',
        '"': ' ',
        "'": ' ',
        '(': ' ',
        ')': ' ',
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    text = re.sub(' +', ' ', text)
    text = re.sub(r'([.!?])', r'\1 ', text)
    
    return text.strip()

def text_to_speech_elevenlabs(text, voice_id=None, model_id="eleven_monolingual_v1"):
    """
    Convert text to speech using ElevenLabs API (primary)
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ElevenLabs API key not found in environment variables")
    
    # Normalize text for TTS
    normalized_text = normalize_text_for_tts(text)
    print(f"ElevenLabs - Normalized text: {normalized_text}")
    
    # Use default voice if not specified
    if not voice_id:
        voice_id = ELEVENLABS_VOICE_ID
    
    # Construct the API endpoint
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
    
    # Set up headers and data
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    data = {
        "text": normalized_text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        # Make the API request with timeout
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Return the audio content
            return response.content
        else:
            error_msg = f"ElevenLabs API error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)
            
    except requests.exceptions.Timeout:
        raise Exception("ElevenLabs API request timed out")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to ElevenLabs API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"ElevenLabs API request failed: {str(e)}")

def text_to_speech_speechify(text, voice_id=None, output_format="mp3"):
    """
    Convert text to speech using Speechify API (secondary)
    """
    if not SPEECHIFY_API_KEY:
        raise ValueError("Speechify API key not found in environment variables")
    
    # Normalize text for TTS
    normalized_text = normalize_text_for_tts(text)
    print(f"Speechify - Normalized text: {normalized_text}")
    
    # Use default voice if not specified
    if not voice_id:
        voice_id = SPEECHIFY_VOICE_ID
    
    # Construct the API endpoint
    url = f"{SPEECHIFY_BASE_URL}/v1/synthesize"
    
    # Set up headers and data
    headers = {
        "Authorization": f"Bearer {SPEECHIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "text": normalized_text,
        "voice": voice_id,
        "format": output_format,
        "speed": 1.0,
        "pitch": 0,
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.content
        else:
            error_msg = f"Speechify API error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)
            
    except requests.exceptions.Timeout:
        raise Exception("Speechify API request timed out")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to Speechify API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Speechify API request failed: {str(e)}")

def text_to_speech_resemble(text, voice_uuid=None, output_format="wav", sample_rate=48000):
    """
    Convert text to speech using Resemble AI API (fallback)
    """
    if not RESEMBLE_AI_API_KEY:
        raise ValueError("Resemble AI API key not found in environment variables")
    
    # Normalize text for TTS
    normalized_text = normalize_text_for_tts(text)
    print(f"Resemble AI - Normalized text: {normalized_text}")
    
    # Use default voice if not specified
    if not voice_uuid:
        voice_uuid = RESEMBLE_AI_VOICE_UUID
    
    url = f"{RESEMBLE_AI_BASE_URL}/synthesize"
    
    headers = {
        "Authorization": f"Bearer {RESEMBLE_AI_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip"
    }
    
    data = {
        "voice_uuid": voice_uuid,
        "data": normalized_text,
        "sample_rate": sample_rate,
        "output_format": output_format
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if "audio_content" in response_data:
                    audio_bytes = base64.b64decode(response_data["audio_content"])
                    return audio_bytes
                else:
                    raise Exception("No audio_content in Resemble AI response")
            except json.JSONDecodeError:
                return response.content
        else:
            error_msg = f"Resemble AI API error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)
            
    except requests.exceptions.Timeout:
        raise Exception("Resemble AI API request timed out")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to Resemble AI API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Resemble AI API request failed: {str(e)}")

def text_to_speech_with_fallback(text):
    """
    Convert text to speech using ElevenLabs first, then Speechify, then Resemble AI as fallback
    """
    print(f"TTS Request: {text[:100]}...")
    
    audio_content = None
    error_messages = []
    
    # Check which TTS providers are configured
    elevenlabs_configured = bool(ELEVENLABS_API_KEY)
    speechify_configured = bool(SPEECHIFY_API_KEY)
    resemble_configured = bool(RESEMBLE_AI_API_KEY)
    
    print(f"TTS Providers - ElevenLabs: {elevenlabs_configured}, Speechify: {speechify_configured}, Resemble: {resemble_configured}")
    
    # Try ElevenLabs first (highest priority)
    if elevenlabs_configured:
        try:
            print("Trying ElevenLabs TTS...")
            audio_content = text_to_speech_elevenlabs(text)
            print("ElevenLabs TTS successful")
            return audio_content, "elevenlabs", None
        except Exception as e:
            error_msg = f"ElevenLabs failed: {str(e)}"
            print(error_msg)
            error_messages.append(error_msg)
    
    # Try Speechify second
    if speechify_configured:
        try:
            print("Trying Speechify TTS...")
            audio_content = text_to_speech_speechify(text)
            print("Speechify TTS successful")
            return audio_content, "speechify", error_messages
        except Exception as e:
            error_msg = f"Speechify failed: {str(e)}"
            print(error_msg)
            error_messages.append(error_msg)
    
    # Try Resemble AI as final fallback
    if resemble_configured:
        try:
            print("Trying Resemble AI TTS (fallback)...")
            audio_content = text_to_speech_resemble(text)
            print("Resemble AI TTS successful")
            return audio_content, "resemble", error_messages
        except Exception as e:
            error_msg = f"Resemble AI failed: {str(e)}"
            print(error_msg)
            error_messages.append(error_msg)
    
    # All providers failed or not configured
    if not elevenlabs_configured and not speechify_configured and not resemble_configured:
        error_messages.append("No TTS providers configured. Please set API keys.")
        try:
            tts = gTTS(text=text, lang='en',slow=False)
            audio_buffer = io.BytesIO()
            tts.writer_to_fpp(audio_buffer)
            audio_buffer.seek(0)
            audio_content = audio_buffer.getValue()
            print("gTTS Success")
            return audio_content, "gTTS", error_messages
        except Exception as e:
            print(f"gTTS fallback failed: {str(e)}")
            return None
    
    print(f"TTS not available. Errors: {error_messages}")
    return None, "none", error_messages

class ChatRequest(BaseModel):
    message: str
    chat_history: list = []
    is_voice: bool = False

class ChatResponse(BaseModel):
    response: str
    chat_history: list = []
    audio_url: str = ""
    tts_provider: str = "none"

user_sessions = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        if "default" not in user_sessions:
            user_sessions["default"] = {
                "input": "",
                "chat_history": [],
                "context": "",
                "response": "",
                "next_step": "generate_response"
            }
        
        current_state = user_sessions["default"]
        current_state["input"] = request.message
        
        current_state["chat_history"].append(HumanMessage(content=request.message))
        
        print(f"\n [Agent is thinking...]")
        
        result = agent_graph.invoke(current_state)
        
        current_state["chat_history"].append(AIMessage(content=result['response']))
        
        user_sessions["default"] = result
        
        audio_url = ""
        tts_provider = "none"
        tts_errors = []
        
        if result['response']:
            try:
                audio_content, tts_provider, tts_errors = text_to_speech_with_fallback(result['response'])
                
                if audio_content:
                    user_sessions["default"]["audio"] = audio_content
                    user_sessions["default"]["tts_provider"] = tts_provider
                    audio_url = f"/audio?t={hash(result['response'])}"
                else:
                    print("TTS not available, continuing without audio")
                    
            except Exception as e:
                print(f"TTS error: {e}")
                tts_errors.append(str(e))
        
        formatted_chat_history = []
        for msg in result['chat_history']:
            if isinstance(msg, HumanMessage):
                formatted_chat_history.append({"sender": "user", "message": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_chat_history.append({"sender": "agent", "message": msg.content})
        
        return ChatResponse(
            response=result['response'],
            chat_history=formatted_chat_history,
            audio_url=audio_url,
            tts_provider=tts_provider
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/audio")
async def get_audio():
    try:
        audio_data = user_sessions["default"].get("audio", b"")
        if not audio_data:
            raise HTTPException(status_code=404, detail="No audio available")
        
        from fastapi import Response
        
        tts_provider = user_sessions["default"].get("tts_provider", "elevenlabs")
        
        # Determine content type based on provider
        if tts_provider == "elevenlabs":
            media_type = "audio/mpeg"
            filename = "response.mp3"
        elif tts_provider == "speechify":
            media_type = "audio/mpeg" 
            filename = "response.mp3"
        else:  # resemble or fallback
            media_type = "audio/wav"
            filename = "response.wav"
        
        return Response(
            content=audio_data,
            media_type=media_type,
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving audio: {str(e)}")

@app.post("/transcribe-audio")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using speech recognition with FFmpeg fallbacks
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            shutil.copyfileobj(audio.file, temp_audio)
            temp_audio_path = temp_audio.name
        
        # Try to convert using FFmpeg (if available)
        wav_path = temp_audio_path.replace(".webm", ".wav")
        conversion_success = False
        
        if check_ffmpeg_installed():
            try:
                # Convert using pydub (requires FFmpeg)
                audio_segment = AudioSegment.from_file(temp_audio_path, format="webm")
                audio_segment.export(wav_path, format="wav")
                conversion_success = True
                print("✅ Audio converted using FFmpeg")
                
            except Exception as conversion_error:
                print(f"⚠️  Pydub conversion failed: {conversion_error}")
                conversion_success = False
        else:
            print("⚠️  FFmpeg not available for conversion")
        
        # Try to transcribe using the appropriate file
        transcription = None
        try:
            recognizer = sr.Recognizer()
            
            if conversion_success:
                # Use the converted WAV file
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    transcription = recognizer.recognize_google(audio_data)
            else:
                # Fallback: try to process the original file directly
                # This may work for some formats without conversion
                try:
                    with sr.AudioFile(temp_audio_path) as source:
                        audio_data = recognizer.record(source)
                        transcription = recognizer.recognize_google(audio_data)
                except:
                    # Final fallback: use the raw binary data
                    audio.file.seek(0)  # Reset file pointer
                    audio_data = sr.AudioData(audio.file.read(), 48000, 2)
                    transcription = recognizer.recognize_google(audio_data)
            
        except sr.UnknownValueError:
            return {"error": "Could not understand the audio"}
        except sr.RequestError as e:
            return {"error": f"Speech recognition error: {e}"}
        except Exception as e:
            return {"error": f"Audio processing error: {str(e)}"}
        
        finally:
            # Clean up temporary files
            Path(temp_audio_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
        
        return {"transcription": transcription}
            
    except Exception as e:
        return {"error": f"Audio processing error: {str(e)}"}

@app.post("/voice-chat")
async def voice_chat_with_agent(request: Request):
    """
    This endpoint should only handle fallback for server environments
    """
    try:
        # Server environment - only provide text fallback
        try:
            request_data = await request.json()
            text_input = request_data.get("text", "")
            
            if text_input:
                # Process as text input
                chat_history = request_data.get("chat_history", [])
                chat_request = ChatRequest(message=text_input, chat_history=chat_history, is_voice=True)
                response = await chat_with_agent(chat_request)
                
                # Add flag to indicate this was a fallback response
                response_dict = response.dict()
                response_dict["is_fallback"] = True
                return JSONResponse(content=response_dict)
        except:
            pass
            
        return JSONResponse(content={
            "error": "Voice input requires client-side speech recognition",
            "message": "Please use a browser that supports speech recognition or type your message.",
            "fallback_available": True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def serve_frontend():
    try:
        with open("securassist.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SecurAssist</title>
        </head>
        <body>
            <h1>SecurAssist</h1>
            <p>HTML file not found. Please check your deployment.</p>
        </body>
        </html>
        """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

@app.get("/debug/env")
async def debug_environment():
    """Check if environment variables are set correctly"""
    return {
        "elevenlabs_api_key_set": bool(ELEVENLABS_API_KEY),
        "speechify_api_key_set": bool(SPEECHIFY_API_KEY),
        "resemble_ai_api_key_set": bool(RESEMBLE_AI_API_KEY),
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "speechify_voice_id": SPEECHIFY_VOICE_ID,
        "resemble_ai_voice_uuid": RESEMBLE_AI_VOICE_UUID,
        "environment": "Render" if os.getenv("RENDER") else "Local",
        "ffmpeg_available": check_ffmpeg_installed()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint that verifies dependencies"""
    status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "ffmpeg_available": check_ffmpeg_installed(),
        "dependencies": {
            "speech_recognition": True,
            "pydub": True,
            "ffmpeg": check_ffmpeg_installed()
        }
    }
    
    # Test each dependency
    try:
        import speech_recognition as sr
    except ImportError:
        status["dependencies"]["speech_recognition"] = False
        status["status"] = "degraded"
    
    try:
        import pydub
    except ImportError:
        status["dependencies"]["pydub"] = False
        status["status"] = "degraded"
    
    if not status["dependencies"]["ffmpeg"]:
        status["status"] = "degraded"
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)