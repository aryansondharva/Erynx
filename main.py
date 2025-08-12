"""
===============================================
🌟 30 Days of AI Voice Agents Challenge
🔧 Backend: FastAPI
🧠 Integrations: Murf AI (TTS) + AssemblyAI (Transcription) + Gemini LLM
===============================================
"""
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import requests
import shutil
import assemblyai as aai
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from io import BytesIO
import requests
import os

# ============================================================
# 🔹 Load API Keys
# ============================================================
load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("✅ Loaded Murf API Key:", bool(MURF_API_KEY))
print("✅ Loaded AssemblyAI Key:", bool(ASSEMBLYAI_API_KEY))
print("✅ Loaded Gemini API Key:", bool(GEMINI_API_KEY))

# Initialize AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

# ============================================================
# 🔹 FastAPI App
# ============================================================
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ============================================================
# 🔹 Murf API TTS (Text → Voice)
# ============================================================
class TextRequest(BaseModel):
    text: str
    voice: str = "default"

@app.post("/generate")
async def generate_voice(data: TextRequest):
    voice_map = {
        "default": "en-US-natalie",
        "narrator": "en-US-terrell",
        "support": "en-US-miles",
        "sergeant": "en-US-ken",
        "game": "en-US-paul"
    }
    voice_id = voice_map.get(data.voice.lower(), "en-US-natalie")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": MURF_API_KEY
    }
    payload = {"text": data.text, "voice_id": voice_id}

    response = requests.post("https://api.murf.ai/v1/speech/generate", headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return JSONResponse(content={"audio_url": result["audioFile"]})
    else:
        return JSONResponse(status_code=response.status_code, content={"error": response.text})

# ============================================================
# 🔹 Audio Upload
# ============================================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_stat = os.stat(file_location)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": file_stat.st_size,
        "message": "🎤 Recording uploaded successfully!",
        "icon": "🎤"
    }

# ============================================================
# 🔹 Transcription Only
# ============================================================
@app.post("/transcribe/file")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        transcript = transcriber.transcribe(audio_bytes)
        return {
            "transcription": transcript.text,
            "status": "🔊 Transcription complete!",
            "icon": "🔊"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================================
# 🔹 Logos
# ============================================================
@app.get("/logo/start")
async def get_start_logo():
    return FileResponse("static/logos/start_recording.png")

@app.get("/logo/microphone")
async def get_microphone_logo():
    return FileResponse("static/logos/microphone.png")

# ============================================================
# 🔹 Direct Transcribe → Murf Voice
# ============================================================
@app.post("/voice-reply")
async def voice_reply(file: UploadFile = File(...), voice: str = Form("default")):
    try:
        audio_bytes = await file.read()
        transcript = transcriber.transcribe(audio_bytes)
        text = transcript.text

        voice_map = {
            "default": "en-US-natalie",
            "narrator": "en-US-terrell",
            "support": "en-US-miles",
            "sergeant": "en-US-ken",
            "game": "en-US-paul"
        }
        voice_id = voice_map.get(voice.lower(), "en-US-natalie")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": MURF_API_KEY
        }
        payload = {"text": text, "voice_id": voice_id, "format": "mp3"}
        murf_response = requests.post("https://api.murf.ai/v1/speech/generate", headers=headers, json=payload)

        if murf_response.status_code != 200:
            return JSONResponse(status_code=murf_response.status_code, content={"error": murf_response.text})

        audio_url = murf_response.json().get("audioFile")
        if not audio_url:
            return JSONResponse(status_code=500, content={"error": "No audio file URL returned by Murf AI"})

        audio_file = requests.get(audio_url)
        if audio_file.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to download audio from Murf AI"})

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/murf-tts")
async def murf_tts_alias(file: UploadFile = File(...), voice: str = Form("default")):
    return await voice_reply(file, voice)

# ============================================================
# 🔹 Text → TTS (JSON)
# ============================================================
class TTSRequest(BaseModel):
    text: str
    voice: str = "default"

@app.post("/murf-tts-json")
async def murf_tts_json(data: TTSRequest):
    try:
        voice_map = {
            "default": "en-US-natalie",
            "narrator": "en-US-terrell",
            "support": "en-US-miles",
            "sergeant": "en-US-ken",
            "game": "en-US-paul"
        }
        voice_id = voice_map.get(data.voice.lower(), "en-US-natalie")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": MURF_API_KEY
        }
        payload = {"text": data.text, "voice_id": voice_id, "format": "mp3"}

        murf_response = requests.post("https://api.murf.ai/v1/speech/generate", headers=headers, json=payload)
        if murf_response.status_code != 200:
            return JSONResponse(status_code=murf_response.status_code, content={"error": murf_response.text})

        audio_url = murf_response.json().get("audioFile")
        if not audio_url:
            return JSONResponse(status_code=500, content={"error": "No audio file URL returned by Murf AI"})

        audio_file = requests.get(audio_url)
        if audio_file.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to download audio from Murf AI"})

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================================
# 🔹 Day 9: Full Non-Streaming Pipeline
# ============================================================
@app.post("/llm/query")
async def llm_query(file: UploadFile = File(...), voice: str = Form("default")):
    """
    Full pipeline: audio -> transcription -> LLM -> Murf TTS -> audio response
    """
    if not GEMINI_API_KEY or not MURF_API_KEY or not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="API keys not configured")

    try:
        audio_bytes = await file.read()
        transcript = transcriber.transcribe(audio_bytes)
        user_text = transcript.text
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        # Gemini call
        gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        payload = {"contents": [{"parts": [{"text": user_text}]}]}

        gemini_response = requests.post(gemini_url, headers=headers, params=params, json=payload)
        gemini_response.raise_for_status()
        data = gemini_response.json()

        if "candidates" not in data or not data["candidates"]:
            raise HTTPException(status_code=500, detail="No response from Gemini API")

        llm_text = data["candidates"][0]["content"]["parts"][0]["text"]
        if len(llm_text) > 3000:
            llm_text = llm_text[:3000]

        # Murf TTS
        voice_map = {
            "default": "en-US-natalie",
            "narrator": "en-US-terrell",
            "support": "en-US-miles",
            "sergeant": "en-US-ken",
            "game": "en-US-paul"
        }
        voice_id = voice_map.get(voice.lower(), "en-US-natalie")

        murf_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": MURF_API_KEY
        }
        murf_payload = {"text": llm_text, "voice_id": voice_id, "format": "mp3"}

        murf_response = requests.post("https://api.murf.ai/v1/speech/generate", headers=murf_headers, json=murf_payload)
        murf_response.raise_for_status()
        murf_data = murf_response.json()

        audio_url = murf_data.get("audioFile")
        if not audio_url:
            raise HTTPException(status_code=500, detail="Murf API did not return audioFile")

        audio_file = requests.get(audio_url)
        audio_file.raise_for_status()

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except requests.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"HTTP Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ==========================================================
# 🔹 Day 10: Chat History Endpoint
# ============================================================
chat_history_store = {}  # { session_id: [ {"role": "user", "content": text}, {"role": "assistant", "content": text} ] }

@app.post("/agent/chat/{session_id}")
async def chat_with_history(session_id: str, file: UploadFile = File(...), voice: str = Form("default")):
    if not GEMINI_API_KEY or not MURF_API_KEY or not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="API keys not configured")

    try:
        # Read audio
        audio_bytes = await file.read()

        # Transcribe using AssemblyAI
        transcript = transcriber.transcribe(audio_bytes)
        user_text = transcript.text
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        # Get or create history
        if session_id not in chat_history_store:
            chat_history_store[session_id] = []
        history = chat_history_store[session_id]

        # Append user message
        history.append({"role": "user", "content": user_text})

        # Prepare Gemini request with full history
        # Gemini API expects alternating user/model roles. Here we map our simple "user" and "assistant" roles.
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        gemini_payload = {"contents": gemini_history}
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}

        gemini_response = requests.post(gemini_url, headers=headers, params=params, json=gemini_payload)
        gemini_response.raise_for_status()
        data = gemini_response.json()

        if "candidates" not in data or not data["candidates"]:
            raise HTTPException(status_code=500, detail="No response from Gemini API")

        llm_text = data["candidates"][0]["content"]["parts"][0]["text"]
        if len(llm_text) > 3000:
            llm_text = llm_text[:3000]

        # Append assistant reply
        history.append({"role": "assistant", "content": llm_text})

        # Murf TTS
        voice_map = {
            "default": "en-US-natalie",
            "narrator": "en-US-terrell",
            "support": "en-US-miles",
            "sergeant": "en-US-ken",
            "game": "en-US-paul"
        }
        voice_id = voice_map.get(voice.lower(), "en-US-natalie")

        murf_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": MURF_API_KEY
        }
        murf_payload = {"text": llm_text, "voice_id": voice_id, "format": "mp3"}

        murf_response = requests.post("https://api.murf.ai/v1/speech/generate", headers=murf_headers, json=murf_payload)
        murf_response.raise_for_status()
        murf_data = murf_response.json()

        audio_url = murf_data.get("audioFile")
        if not audio_url:
            raise HTTPException(status_code=500, detail="Murf API did not return audioFile")

        audio_file = requests.get(audio_url)
        audio_file.raise_for_status()

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except requests.HTTPError as e:
        # Check for 400 Bad Request which may indicate a history formatting issue
        if e.response.status_code == 400 and "contents" in e.response.text:
            error_message = "Gemini API error: The conversation history might be improperly formatted or have alternating roles. Please check the `gemini_history` structure."
            raise HTTPException(status_code=400, detail=error_message) from e
        raise HTTPException(status_code=500, detail=f"HTTP Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")