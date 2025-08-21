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

# Initialize TTS APIs
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

def text_to_speech_speechify(text, voice_id=None, output_format="mp3"):
    """
    Convert text to speech using Speechify API (primary)
    """
    if not SPEECHIFY_API_KEY:
        raise ValueError("Speechify API key not found in environment variables")
    
    # Normalize text for TTS
    normalized_text = normalize_text_for_tts(text)
    print(f"Speechify - Normalized text: {normalized_text}")
    
    # Use default voice if not specified
    if not voice_id:
        voice_id = SPEECHIFY_VOICE_ID
    
    # Construct the API endpoint (adjust based on Speechify's actual API)
    url = f"{SPEECHIFY_BASE_URL}/v1/synthesize"
    
    # Set up headers and data (adjust based on Speechify's actual API requirements)
    headers = {
        "Authorization": f"Bearer {SPEECHIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "text": normalized_text,
        "voice": voice_id,
        "format": output_format,
        "speed": 1.0,  # Adjust as needed
        "pitch": 0,    # Adjust as needed
    }
    
    try:
        # Make the API request with timeout
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Return the audio content (adjust based on Speechify's response format)
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
    
    # Construct the API endpoint
    url = f"{RESEMBLE_AI_BASE_URL}/synthesize"
    
    # Set up headers and data
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
        # Make the API request with timeout
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Try to parse as JSON first
            try:
                response_data = response.json()
                if "audio_content" in response_data:
                    # Decode the base64 audio content
                    audio_bytes = base64.b64decode(response_data["audio_content"])
                    return audio_bytes
                else:
                    raise Exception("No audio_content in Resemble AI response")
            except json.JSONDecodeError:
                # If response is not JSON, assume it's raw audio
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

def text_to_speech_with_fallback(text, tts_provider="speechify"):
    """
    Convert text to speech using Speechify first, fallback to Resemble AI if it fails
    """
    audio_content = None
    used_provider = tts_provider
    error_messages = []
    
    # Try Speechify first
    if tts_provider == "speechify" and SPEECHIFY_API_KEY:
        try:
            print("Trying Speechify TTS...")
            audio_content = text_to_speech_speechify(text)
            print("Speechify TTS successful")
            return audio_content, "speechify", None
        except Exception as e:
            error_msg = f"Speechify failed: {str(e)}"
            print(error_msg)
            error_messages.append(error_msg)
            # Continue to fallback
    
    # Try Resemble AI as fallback
    if RESEMBLE_AI_API_KEY:
        try:
            print("Trying Resemble AI TTS (fallback)...")
            audio_content = text_to_speech_resemble(text)
            print("Resemble AI TTS successful")
            return audio_content, "resemble", error_messages
        except Exception as e:
            error_msg = f"Resemble AI failed: {str(e)}"
            print(error_msg)
            error_messages.append(error_msg)
    
    # Both providers failed
    return None, "none", error_messages

class ChatRequest(BaseModel):
    message: str
    chat_history: list = []

class ChatResponse(BaseModel):
    response: str
    chat_history: list = []
    audio_url: str = ""
    tts_provider: str = "none"  # Add provider info to response

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
        if result['response']:
            try:
                # Generate audio using Speechify with Resemble fallback
                audio_content, tts_provider, errors = text_to_speech_with_fallback(result['response'])
                
                if audio_content:
                    # Store audio in session
                    user_sessions["default"]["audio"] = audio_content
                    user_sessions["default"]["tts_provider"] = tts_provider
                    audio_url = f"/audio?t={hash(result['response'])}"
                else:
                    print("All TTS providers failed. Errors:", errors)
                    
            except Exception as e:
                print(f"TTS error: {e}")
        
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
        
        # Determine content type based on provider
        tts_provider = user_sessions["default"].get("tts_provider", "resemble")
        media_type = "audio/wav" if tts_provider == "resemble" else "audio/mp3"
        filename = "response.wav" if tts_provider == "resemble" else "response.mp3"
        
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
        # Check if we're in a server environment (no microphone access)
        is_server_environment = os.getenv("RENDER", False) or os.getenv("PYTHONANYWHERE", False) or os.getenv("HEROKU", False)
        
        if is_server_environment:
            # In server environment, return instructions for voice input
            return JSONResponse(content={
                "error": "Voice input not available on server",
                "message": "Please use text input instead. Voice chat only works in local environments with microphone access."
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
        # Try to read the HTML file
        with open("securassist.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        # If file doesn't exist, serve a basic version
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SecurAssist</title>
            <!-- Your CSS styles here -->
        </head>
        <body>
            <h1>SecurAssist</h1>
            <p>HTML file not found. Please check your deployment.</p>
        </body>
        </html>
        """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

@app.get("/tts-status")
async def tts_status():
    """Endpoint to check TTS provider status"""
    return {
        "speechify_configured": bool(SPEECHIFY_API_KEY),
        "resemble_ai_configured": bool(RESEMBLE_AI_API_KEY),
        "active_provider": user_sessions["default"].get("tts_provider", "none") if "default" in user_sessions else "none"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)