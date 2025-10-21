from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx, sqlite3, json
from datetime import datetime
import os

# === CONFIG ===
API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = os.getenv("DB_PATH", "/data/negotiations.db")  # your Render disk is mounted at /data

# === APP ===
app = FastAPI()

# CORS (loose for now; you can tighten allow_origins to your site later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DB helpers ===
def get_conn():
    # SQLite is a file DB; this opens the same path each time
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_schema():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    c = conn.cursor()

    # Base table creation
    c.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            transcript   TEXT NOT NULL
            -- (we'll add student_name and feedback below for backward compat)
        )
    """)
    # Add columns if the table pre-existed without them
    c.execute("PRAGMA table_info(transcripts)")
    cols = {row[1] for row in c.fetchall()}
    if "student_name" not in cols:
        c.execute("ALTER TABLE transcripts ADD COLUMN student_name TEXT")
    if "feedback" not in cols:
        c.execute("ALTER TABLE transcripts ADD COLUMN feedback TEXT")

    # Helpful index
    c.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_student ON transcripts(student_id)")
    conn.commit()
    conn.close()

ensure_schema()

# === Routes ===
@app.get("/")
def root():
    return {"status": "ok", "message": "Negotiation backend is running!"}

@app.post("/message")
async def message(request: Request):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    data = await request.json()
    student_id = data.get("student_id", "anon")
    transcript = data.get("transcript", [])
    user_msg = data.get("message")
    # (student_name is optional here; usually not provided until feedback)
    student_name = data.get("student_name")

    if user_msg is None:
        raise HTTPException(status_code=400, detail="Missing 'message' field")

    # 1) Append user message locally
    transcript.append({"role": "user", "content": user_msg})

    # 2) Call OpenRouter chat completions
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": transcript
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e.response.text[:300]}")

    resp_json = r.json()
    try:
        bot_msg = resp_json["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {resp_json}")

    # 3) Append assistant message & persist this turn
    transcript.append(bot_msg)

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO transcripts (student_id, student_name, timestamp, transcript) VALUES (?,?,?,?)",
        (student_id, student_name, datetime.utcnow().isoformat(), json.dumps(transcript))
    )
    conn.commit()
    conn.close()

    return {"reply": bot_msg.get("content", ""), "transcript": transcript}

@app.post("/feedback")
async def feedback(request: Request):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    data = await request.json()
    student_id = data.get("student_id", "anon")
    student_name = (data.get("student_name") or "").strip()
    transcript = data.get("transcript")

    if not transcript:
        raise HTTPException(status_code=400, detail="Missing 'transcript' field")
    if not student_name:
        # Frontend should have forced name entry; enforce on backend too
        raise HTTPException(status_code=400, detail="Missing 'student_name' (name required for feedback)")

    # 1) Call OpenRouter for feedback text
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a negotiation coach. Give constructive, specific, and actionable feedback."},
            {"role": "user", "content": f"Here is a transcript:\n{json.dumps(transcript, ensure_ascii=False)}"}
        ]
    }

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e.response.text[:300]}")

    resp_json = r.json()
    try:
        feedback_text = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {resp_json}")

    # 2) Append the feedback into the transcript so it's part of the record
    transcript.append({"role": "feedback", "content": feedback_text})

    # 3) Persist a final row with name + feedback + full transcript
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO transcripts (student_id, student_name, timestamp, transcript, feedback) VALUES (?,?,?,?,?)",
        (student_id, student_name, datetime.utcnow().isoformat(), json.dumps(transcript), feedback_text)
    )
    conn.commit()
    conn.close()

    # 4) Return feedback (frontend already displays it with class "feedback")
    return {"feedback": feedback_text, "transcript": transcript}

# Optional: download the raw SQLite file (use carefully; remove for public)
@app.get("/download_db")
def download_db(secret: str | None = None):
    allowed = os.getenv("DOWNLOAD_KEY")  # set in Render → Environment if you want protection
    if allowed and secret != allowed:
        raise HTTPException(status_code=403, detail="unauthorized")
    return FileResponse(DB_PATH, filename="negotiations.db")
