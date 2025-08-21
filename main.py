from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

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
    
    print(f"TTS not available. Errors: {error_messages}")
    return None, "none", error_messages

class ChatRequest(BaseModel):
    message: str
    chat_history: list = []

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

@app.post("/voice-chat")
async def voice_chat_with_agent(request: Request):
    try:
        # Check if we're in a server environment
        is_server_environment = os.getenv("RENDER", False) or os.getenv("SERVER", False)
        
        if is_server_environment:
            # For server environments, provide a text input fallback
            try:
                request_data = await request.json()
                text_input = request_data.get("text", "")
                
                if text_input:
                    # Process as text input instead of voice
                    chat_history = request_data.get("chat_history", [])
                    chat_request = ChatRequest(message=text_input, chat_history=chat_history)
                    response = await chat_with_agent(chat_request)
                    return response
                else:
                    return JSONResponse(content={
                        "error": "Voice input not available on server",
                        "message": "Please use the text input field or provide text in the request body.",
                        "fallback_available": True
                    })
                    
            except Exception as json_error:
                return JSONResponse(content={
                    "error": "Voice input not available on server",
                    "message": "Please use the text input field.",
                    "fallback_available": False
                })
        
        # Local environment with microphone access
        user_input = ""
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            try:
                user_input = recognizer.recognize_google(audio)
                print("You said: ", user_input)
            except sr.WaitTimeoutError:
                return JSONResponse(content={"error": "No speech detected"})
            except sr.UnknownValueError:
                return JSONResponse(content={"error": "Couldn't understand audio"})
            except sr.RequestError:
                return JSONResponse(content={"error": "Could not request results from Google Speech Recognition service"})
        
        if not user_input:
            return JSONResponse(content={"error": "No input received"})

        # Get chat history from request if available
        try:
            request_data = await request.json()
            chat_history = request_data.get("chat_history", [])
        except:
            chat_history = []
        
        # Process the voice input
        chat_request = ChatRequest(message=user_input, chat_history=chat_history)
        response = await chat_with_agent(chat_request)
        
        return response
        
    except Exception as e:
        print(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing voice request: {str(e)}")

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
        "environment": "Render" if os.getenv("RENDER") else "Local"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)