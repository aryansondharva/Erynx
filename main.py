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
import json
from pathlib import Path

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
app = FastAPI(title="N9NE AI Voice Agent")

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
# 🔹 Fallback audio utilities (Day 11)
# ============================================================
FALLBACK_AUDIO_CANDIDATES = [
    "static/audio/fallback.mp3",  # optional if you add one
    "static/audio/282ee83b-dfa3-4c52-8a7d-5113884c9b17.mp3",
    "static/audio/4dd33907-e332-4435-8289-f427f527e783.mp3",
    "static/audio/61e566e2-c60c-4111-a37b-6ac40ea92ea7.mp3",
    "static/audio/6e4c10a8-6cd6-4d76-9246-1a2001aea43d.mp3",
    "static/audio/897f3d65-908e-41f6-afb7-b6f8021cc5a9.mp3",
    "static/audio/8ea6a778-4a56-4fc5-9132-16aae5032f8a.mp3",
    "static/audio/db2a9484-a00c-4acd-a26a-08870cbd3d7f.mp3",
    "static/audio/e4e571f4-d615-4c64-b8c2-aaf3de318ba3.mp3",
    "static/audio/f32daac2-5143-4513-a1e6-ec8cf7fb8e53.mp3",
    "static/audio/f67ae3f4-5c99-4458-a40f-4119a23ea130.mp3",
    "static/audio/fa69592b-d854-42e8-a40c-6ce1d568fb86.mp3",
    "sample_voice.mp3",
]

FALLBACK_MESSAGE = "I'm having trouble connecting right now. Please try again in a moment."


def get_fallback_audio_bytes() -> bytes:
    for path in FALLBACK_AUDIO_CANDIDATES:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
        except Exception:
            continue
    # As a last resort, return empty bytes (client should handle)
    return b""


@app.get("/fallback/audio")
async def fallback_audio():
    audio_bytes = get_fallback_audio_bytes()
    if not audio_bytes:
        return JSONResponse(status_code=503, content={"error": "No fallback audio available"})
    return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

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
        "api-key": MURF_API_KEY or ""
    }
    payload = {"text": data.text, "voice_id": voice_id}

    try:
        response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json()
            return JSONResponse(content={"audio_url": result.get("audioFile")})
        else:
            # Fallback on error
            audio_bytes = get_fallback_audio_bytes()
            return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        audio_bytes = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

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
# 🔹 Transcription Only (with error handling)
# ============================================================
@app.post("/transcribe/file")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not ASSEMBLYAI_API_KEY:
            raise RuntimeError("AssemblyAI key missing")
        transcript = transcriber.transcribe(audio_bytes)
        return {
            "transcription": transcript.text,
            "status": "🔊 Transcription complete!",
            "icon": "🔊"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "message": "Transcription failed. Please try again.",
        })

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
# 🔹 Direct Transcribe → Murf Voice (with fallback)
# ============================================================
@app.post("/voice-reply")
async def voice_reply(file: UploadFile = File(...), voice: str = Form("default")):
    try:
        audio_bytes = await file.read()
        if not ASSEMBLYAI_API_KEY:
            raise RuntimeError("AssemblyAI key missing")
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
            "api-key": MURF_API_KEY or ""
        }
        payload = {"text": text, "voice_id": voice_id, "format": "mp3"}
        murf_response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if murf_response.status_code != 200:
            raise RuntimeError("Murf TTS failed")

        audio_url = murf_response.json().get("audioFile")
        if not audio_url:
            raise RuntimeError("No audio file URL returned by Murf AI")

        audio_file = requests.get(audio_url, timeout=30)
        audio_file.raise_for_status()

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except Exception:
        # Fallback audio
        audio_bytes = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

@app.post("/murf-tts")
async def murf_tts_alias(file: UploadFile = File(...), voice: str = Form("default")):
    return await voice_reply(file, voice)

# ============================================================
# 🔹 Text → TTS (JSON) (with fallback)
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
            "api-key": MURF_API_KEY or ""
        }
        payload = {"text": data.text, "voice_id": voice_id, "format": "mp3"}

        murf_response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=headers,
            json=payload,
            timeout=30,
        )
        murf_response.raise_for_status()
        audio_url = murf_response.json().get("audioFile")
        if not audio_url:
            raise RuntimeError("No audio file URL returned by Murf AI")

        audio_file = requests.get(audio_url, timeout=30)
        audio_file.raise_for_status()

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except Exception:
        audio_bytes = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

# ============================================================
# 🔹 Day 9: Full Non-Streaming Pipeline (with fallback)
# ============================================================
@app.post("/llm/query")
async def llm_query(file: UploadFile = File(...), voice: str = Form("default")):
    """
    Full pipeline: audio -> transcription -> LLM -> Murf TTS -> audio response
    """
    if not GEMINI_API_KEY or not MURF_API_KEY or not ASSEMBLYAI_API_KEY:
        # Directly return fallback audio when missing keys
        audio_bytes = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

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

        gemini_response = requests.post(gemini_url, headers=headers, params=params, json=payload, timeout=60)
        gemini_response.raise_for_status()
        data = gemini_response.json()

        if "candidates" not in data or not data["candidates"]:
            raise RuntimeError("No response from Gemini API")

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
            "api-key": MURF_API_KEY or ""
        }
        murf_payload = {"text": llm_text, "voice_id": voice_id, "format": "mp3"}

        murf_response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=murf_headers,
            json=murf_payload,
            timeout=60,
        )
        murf_response.raise_for_status()
        murf_data = murf_response.json()

        audio_url = murf_data.get("audioFile")
        if not audio_url:
            raise RuntimeError("Murf API did not return audioFile")

        audio_file = requests.get(audio_url, timeout=60)
        audio_file.raise_for_status()

        return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")

    except Exception:
        audio_bytes = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

# ==========================================================
# 🔹 Day 10 + 11: Chat History with Permanent Storage + Fallbacks
# ============================================================
# File storage for chat history
CHAT_HISTORY_FILE = "chat_history.json"
chat_history_store = {}

def load_chat_history():
    """Load chat history from JSON file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded chat history: {len(data)} sessions")
                return data
        else:
            print("📝 No existing chat history file found, starting fresh")
            return {}
    except Exception as e:
        print(f"⚠️ Error loading chat history: {e}")
        return {}

def save_chat_history():
    """Save chat history to JSON file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_history_store, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved chat history: {len(chat_history_store)} sessions")
    except Exception as e:
        print(f"❌ Error saving chat history: {e}")

# Load existing chat history on startup
chat_history_store = load_chat_history()

@app.post("/agent/chat/{session_id}")
async def chat_with_history(session_id: str, file: UploadFile = File(...), voice: str = Form("default")):
    # If any critical key is missing, return fallback immediately
    if not ASSEMBLYAI_API_KEY or not GEMINI_API_KEY or not MURF_API_KEY:
        # Append fallback assistant message to history for transparency
        if session_id not in chat_history_store:
            chat_history_store[session_id] = []
        chat_history_store[session_id].append({"role": "assistant", "content": FALLBACK_MESSAGE})
        save_chat_history()
        audio_bytes = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

    try:
        # Read audio
        audio_bytes = await file.read()

        # Transcribe using AssemblyAI
        try:
            transcript = transcriber.transcribe(audio_bytes)
            user_text = transcript.text
        except Exception:
            # Transcription failed → fallback
            if session_id not in chat_history_store:
                chat_history_store[session_id] = []
            chat_history_store[session_id].append({"role": "assistant", "content": FALLBACK_MESSAGE})
            save_chat_history()
            fb = get_fallback_audio_bytes()
            return StreamingResponse(BytesIO(fb), media_type="audio/mpeg")

        if not user_text or not user_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        # Get or create history
        if session_id not in chat_history_store:
            chat_history_store[session_id] = []
        history = chat_history_store[session_id]

        # Append user message
        history.append({"role": "user", "content": user_text})
        save_chat_history()

        # Prepare Gemini request with full history
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

        gemini_payload = {"contents": gemini_history}
        gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}

        try:
            gemini_response = requests.post(
                gemini_url, headers=headers, params=params, json=gemini_payload, timeout=60
            )
            gemini_response.raise_for_status()
            data = gemini_response.json()
            if "candidates" not in data or not data["candidates"]:
                raise RuntimeError("No response from Gemini API")
            llm_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            # LLM failure → fallback
            history.append({"role": "assistant", "content": FALLBACK_MESSAGE})
            save_chat_history()
            fb = get_fallback_audio_bytes()
            return StreamingResponse(BytesIO(fb), media_type="audio/mpeg")

        if len(llm_text) > 3000:
            llm_text = llm_text[:3000]

        # Append assistant reply
        history.append({"role": "assistant", "content": llm_text})
        save_chat_history()

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
            "api-key": MURF_API_KEY or ""
        }
        murf_payload = {"text": llm_text, "voice_id": voice_id, "format": "mp3"}

        try:
            murf_response = requests.post(
                "https://api.murf.ai/v1/speech/generate",
                headers=murf_headers,
                json=murf_payload,
                timeout=60,
            )
            murf_response.raise_for_status()
            murf_data = murf_response.json()
            audio_url = murf_data.get("audioFile")
            if not audio_url:
                raise RuntimeError("Murf API did not return audioFile")
            audio_file = requests.get(audio_url, timeout=60)
            audio_file.raise_for_status()
            return StreamingResponse(BytesIO(audio_file.content), media_type="audio/mpeg")
        except Exception:
            fb = get_fallback_audio_bytes()
            return StreamingResponse(BytesIO(fb), media_type="audio/mpeg")

    except HTTPException as e:
        raise e
    except Exception as e:
        fb = get_fallback_audio_bytes()
        return StreamingResponse(BytesIO(fb), media_type="audio/mpeg")

@app.get("/agent/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session (for debugging)"""
    if session_id not in chat_history_store:
        return {"messages": [], "session_id": session_id}
    
    return {
        "messages": chat_history_store[session_id],
        "session_id": session_id,
        "message_count": len(chat_history_store[session_id])
    }

@app.delete("/agent/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a specific chat session"""
    if session_id in chat_history_store:
        del chat_history_store[session_id]
        save_chat_history()
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/agent/chat/all")
async def delete_all_chat_sessions():
    """Delete all chat sessions"""
    session_count = len(chat_history_store)
    chat_history_store.clear()
    save_chat_history()
    return {"message": f"All {session_count} sessions deleted successfully"}

@app.get("/agent/chat/sessions/list")
async def list_all_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, messages in chat_history_store.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(messages),
            "last_message": messages[-1]["content"][:100] + "..." if messages else "No messages",
            "created": "N/A"
        })
    return {"sessions": sessions, "total_sessions": len(sessions)}

@app.get("/agent/chat/test")
async def test_chat_endpoint():
    """Test endpoint to verify chat history processing works"""
    return {
        "status": "working",
        "message": "Chat history endpoint is functional",
        "chat_history_file": CHAT_HISTORY_FILE,
        "sessions_count": len(chat_history_store),
        "api_keys": {
            "gemini": bool(GEMINI_API_KEY),
            "murf": bool(MURF_API_KEY),
            "assemblyai": bool(ASSEMBLYAI_API_KEY)
        }
    }

# Graceful shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Save chat history when server shuts down"""
    print("🔄 Server shutting down, saving chat history...")
    save_chat_history()
    print("✅ Chat history saved successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)