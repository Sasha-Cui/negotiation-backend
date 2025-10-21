from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx, sqlite3, json
from datetime import datetime
import os
API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = os.getenv("DB_PATH", "/data/negotiations.db")

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    timestamp TEXT,
                    transcript TEXT
                 )""")
    conn.commit()
    conn.close()

init_db()

@app.get("/")
def root():
    return {"status": "ok", "message": "Negotiation backend is running!"}


######## code for downloading db ########
@app.get("/download_db")
def download_db():
    return FileResponse(DB_PATH, filename="negotiations.db")
######## code for downloading db ########

@app.post("/message")
async def message(request: Request):
    data = await request.json()
    student_id = data.get("student_id", "anon")
    transcript = data.get("transcript", [])
    user_msg = data["message"]

    transcript.append({"role": "user", "content": user_msg})

    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"model": "openai/gpt-4o-mini", "messages": transcript}

    async with httpx.AsyncClient() as client:
        r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload)

    bot_msg = r.json()["choices"][0]["message"]
    transcript.append(bot_msg)

    conn = sqlite3.connect("/data/negotiations.db")
    c = conn.cursor()
    c.execute("INSERT INTO transcripts (student_id, timestamp, transcript) VALUES (?,?,?)",
              (student_id, datetime.utcnow().isoformat(), json.dumps(transcript)))
    conn.commit()
    conn.close()

    return {"reply": bot_msg["content"], "transcript": transcript}


@app.post("/feedback")
async def feedback(request: Request):
    data = await request.json()
    transcript = data["transcript"]

    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a negotiation coach. Give constructive feedback."},
            {"role": "user", "content": f"Here is a transcript:\n{transcript}"}
        ]
    }

    async with httpx.AsyncClient() as client:
        r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload)

    return {"feedback": r.json()["choices"][0]["message"]["content"]}
