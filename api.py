"""
api.py
-------
Milestone 2 - Task 6: Processing API & Service Integration

Exposes the full pipeline (pipeline.py) as an HTTP API, so the same
Upload -> Whisper -> Transcript -> LLM Processing -> Summary ->
Action Extraction -> Participant Mapping -> Database flow used by the
Streamlit UI can also be called from other services / a future frontend.

Run with:
    uvicorn api:app --reload

Endpoints:
    POST /process-meeting         upload a recording, run the full pipeline
    GET  /meetings                list all processed meetings
    GET  /meetings/{meeting_id}   full detail: transcript + summary + action items
    GET  /health                  liveness check
"""

import os
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()  # picks up GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY from a local .env

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import process_meeting_recording, PipelineError
from database import Database

app = FastAPI(
    title="AI-Powered Career Intelligence Platform - Meeting Processing API",
    description="Milestone 1 (Audio Processing & Transcription) + Milestone 2 (Summarization & Action Extraction)",
    version="1.0.0",
)

db = Database()


class ActionItemOut(BaseModel):
    task: str
    assignee: str
    deadline: str
    priority: str
    status: str


class ProcessMeetingResponse(BaseModel):
    meeting_id: int
    language: str
    audio_duration_seconds: float
    summary: str
    key_points: list
    decisions: list
    priorities: list
    action_items: list
    participants: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-meeting", response_model=ProcessMeetingResponse)
async def process_meeting(file: UploadFile = File(...)):
    """
    Task 6 flow, exposed over HTTP:
    Upload Meeting -> Whisper -> Transcript -> LLM Processing -> Summary ->
    Action Extraction -> Participant Mapping -> Database.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = os.path.join(tmp_dir, file.filename)
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(raw_path)

        try:
            result = process_meeting_recording(
                raw_file_path=raw_path,
                original_filename=file.filename,
                file_size_bytes=file_size,
                db=db,
            )
        except PipelineError as e:
            # Report exactly which stage failed - much more actionable
            # for API consumers than a generic 500.
            raise HTTPException(status_code=422, detail={"stage": e.stage, "message": str(e)})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

        return ProcessMeetingResponse(
            meeting_id=result.meeting_id,
            language=result.language or "unknown",
            audio_duration_seconds=result.audio_duration_seconds or 0.0,
            summary=result.summary.summary,
            key_points=result.summary.key_points,
            decisions=result.summary.decisions,
            priorities=result.summary.priorities,
            action_items=[
                {
                    "task": entry["action_item"].task,
                    "assignee": entry["participant"].name,
                    "deadline": entry["action_item"].deadline,
                    "priority": entry["action_item"].priority,
                    "status": entry["action_item"].status,
                }
                for entry in result.action_items
            ],
            participants=[p.name for p in result.participants],
        )


@app.get("/meetings")
def list_meetings():
    return db.list_meetings()


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: int):
    data = db.get_meeting_full(meeting_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
    return JSONResponse(content=data)
