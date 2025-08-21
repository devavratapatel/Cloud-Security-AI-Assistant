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

app = FastAPI(title="SecurAssist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
recognizer = sr.Recognizer()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") 
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

def normalize_text_for_tts(text):
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
    if not ELEVENLABS_API_KEY:
        raise ValueError("ElevenLabs API key not found in environment variables")
    
    normalized_text = normalize_text_for_tts(text)
    print(f"Normalized for TTS: {normalized_text}")
    
    if not voice_id:
        voice_id = ELEVENLABS_VOICE_ID 
    

    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
    
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

    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")

class ChatRequest(BaseModel):
    message: str
    chat_history: list = []

class ChatResponse(BaseModel):
    response: str
    chat_history: list = []
    audio_url: str = ""

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
        if result['response']:
            try:
                audio_content = text_to_speech_elevenlabs(result['response'])
                
                user_sessions["default"]["audio"] = audio_content
                audio_url = f"/audio?t={hash(result['response'])}"  
            except Exception as e:
                print(f"ElevenLabs TTS error: {e}")
        
        formatted_chat_history = []
        for msg in result['chat_history']:
            if isinstance(msg, HumanMessage):
                formatted_chat_history.append({"sender": "user", "message": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_chat_history.append({"sender": "agent", "message": msg.content})
        
        return ChatResponse(
            response=result['response'],
            chat_history=formatted_chat_history,
            audio_url=audio_url
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
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving audio: {str(e)}")

@app.post("/voice-chat")
async def voice_chat_with_agent():
    try:
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

        request = ChatRequest(message=user_input)
        response = await chat_with_agent(request)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice request: {str(e)}")

@app.get("/")
async def serve_frontend():
    try:
        with open("securassist.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        # Fallback: return a simple message or create the HTML on the fly
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)