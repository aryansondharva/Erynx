# Erynx - System Architecture

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   User Voice     │      │   Web Audio API   │      │  FastAPI Server  │
│     Input        │─────▶│    (Recording)    │─────▶│    (Backend)     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
          │                         │                         │
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  AssemblyAI STT  │      │  Google Gemini   │      │    Murf AI       │
│(Speech-to-Text)  │      │      (LLM)       │      │(Text-to-Speech)  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
          │                         │                         │
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Transcription   │      │   AI Response    │      │  Audio Stream    │
│  + Confidence    │      │   + Context      │      │   + Fallback     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Component Description

### Input Layer
- **User Voice Input**: Raw voice input from user
- **Web Audio API**: Handles audio recording in browser
- **FastAPI Server**: Backend service processing requests

### Processing Layer
- **AssemblyAI STT**: Converts speech to text with confidence scores
- **Google Gemini**: Large Language Model for processing and response
- **Murf AI**: Converts text responses to natural speech

### Output Layer
- **Transcription**: Text with confidence metrics
- **AI Response**: Contextual responses from LLM
- **Audio Stream**: Speech output with fallback options

## Data Flow

1. User speaks into the microphone
2. Web Audio API captures and processes the audio
3. Audio is sent to FastAPI backend
4. Backend processes audio through three parallel paths:
   - Speech-to-text conversion
   - Language understanding and response generation
   - Text-to-speech synthesis
5. Results are combined and sent back to user

## Technical Details

- **Audio Format**: WebM (Browser Recording)
- **API Communication**: REST/HTTP
- **Response Format**: Streaming MP3 Audio
- **Session Management**: Unique session IDs
- **Data Persistence**: JSON-based chat history
