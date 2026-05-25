from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

load_dotenv()

app = FastAPI(title="Groq Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory session store  { session_id: [messages] }
sessions: dict[str, list[dict]] = {}

SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable assistant.
Answer clearly and concisely. If you don't know something, say so."""


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str ="llama-3.1-8b-instant"   # change this line   # fast & free on Groq
    temperature: float = 0.7

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str
    message_count: int
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    history: list[dict]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Groq Chatbot API is running 🚀", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message and get a reply. Maintains conversation history per session."""
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file")

    # Create or resume a session
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model=req.model,
            temperature=req.temperature,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        )
    except Exception as e:
        print(f"❌ GROQ ERROR: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Groq error: {type(e).__name__}: {str(e)}")

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        model=req.model,
        message_count=len(history),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/session/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    """Retrieve full conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    history = sessions[session_id]
    return SessionInfo(session_id=session_id, message_count=len(history), history=history)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clear / reset a conversation session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted"}


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return {
        "active_sessions": len(sessions),
        "sessions": [
            {"session_id": sid, "message_count": len(msgs)}
            for sid, msgs in sessions.items()
        ],
    }