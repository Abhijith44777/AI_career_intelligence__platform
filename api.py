"""
api.py
-------
Milestone 2 - Task 6: Processing API & Service Integration
Milestone 3 - Meeting Knowledge Repository, Semantic Search & RAG Q&A

Exposes the full pipeline (pipeline.py) as an HTTP API, so the same
Upload -> Whisper -> Transcript -> LLM Processing -> Summary ->
Action Extraction -> Participant Mapping -> Database flow used by the
Streamlit UI can also be called from other services / a future frontend,
plus Milestone 3's knowledge repository, embedding, vector search and RAG
endpoints.

Run with:
    uvicorn api:app --reload

Milestone 1 + 2 endpoints (unchanged):
    POST /process-meeting                     upload a recording, run the full pipeline
    GET  /meetings                            list all processed meetings
    GET  /meetings/{meeting_id}                full detail: transcript + summary + action items
    GET  /health                              liveness check

Milestone 3 endpoints (new):
    GET  /meetings/{meeting_id}/transcript     Task 1 - just the transcript
    GET  /meetings/{meeting_id}/summary        Task 1 - just the summary
    GET  /meetings/{meeting_id}/decisions      Task 1 - just the decisions
    GET  /meetings/{meeting_id}/action-items   Task 1 - just the action items
    GET  /meetings/{meeting_id}/participants   Task 1 - just the participants
    GET  /meetings/{meeting_id}/deadlines      Task 1 - just the deadlines
    GET  /meetings/{meeting_id}/verify         Task 1 - retrieval/linkage verification report
    GET  /meetings/{meeting_id}/vectors        Task 3 - every vector stored for this meeting
    POST /embeddings/generate                  Task 2/3 - (re)index one meeting
    POST /embeddings/sync                      Task 2/3 - reindex every meeting (data sync)
    DELETE /embeddings/{meeting_id}             Task 3 - remove a meeting's vectors
    POST /search/meetings                      Task 4 - semantic search
    POST /ask/meetings                         Task 5 - RAG question answering
"""

import os
import shutil
import tempfile
import logging

from dotenv import load_dotenv
load_dotenv()  # picks up GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY / EMBEDDING_* from a local .env

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                     format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import process_meeting_recording, PipelineError
from database import Database

# Milestone 3
from embedding_service import EmbeddingService
from vector_store import VectorStore
from knowledge_repository import index_meeting, reindex_all_meetings, verify_meeting_retrieval
from semantic_search import semantic_search
from rag_qa import answer_question

app = FastAPI(
    title="AI-Powered Career Intelligence Platform - Meeting Processing API",
    description=(
        "Milestone 1 (Audio Processing & Transcription) + "
        "Milestone 2 (Summarization & Action Extraction) + "
        "Milestone 3 (Meeting Knowledge Repository, Semantic Search & RAG Q&A)"
    ),
    version="1.1.0",
)

# Singletons - loaded once at process startup, not per-request, so the
# embedding client / vector store connection aren't re-initialized on every
# call (this is the "loading models/services only once" performance
# requirement from Task 4).
db = Database()
embedding_service = EmbeddingService()
vector_store = VectorStore()


def _require_meeting(meeting_id: int) -> dict:
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
    return meeting


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/process-meeting", response_model=ProcessMeetingResponse)
async def process_meeting(file: UploadFile = File(...)):
    """
    Task 6 flow, exposed over HTTP:
    Upload Meeting -> Whisper -> Transcript -> LLM Processing -> Summary ->
    Action Extraction -> Participant Mapping -> Database -> (Milestone 3)
    Embedding Generation -> Vector Database.
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
                embedding_service=embedding_service,
                vector_store=vector_store,
            )
        except PipelineError as e:
            # Report exactly which stage failed - much more actionable
            # for API consumers than a generic 500.
            logger.error("Pipeline failed at stage=%s: %s", e.stage, e)
            raise HTTPException(status_code=422, detail={"stage": e.stage, "message": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error while processing meeting")
            raise HTTPException(status_code=500, detail="Unexpected error while processing the meeting.")

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


# ===========================================================================
# Milestone 3 - Task 1: Meeting Knowledge Repository (granular endpoints)
# ===========================================================================

@app.get("/meetings/{meeting_id}/transcript")
def get_meeting_transcript(meeting_id: int):
    _require_meeting(meeting_id)
    return {"meeting_id": meeting_id, "transcript": db.get_transcript(meeting_id)}


@app.get("/meetings/{meeting_id}/summary")
def get_meeting_summary(meeting_id: int):
    _require_meeting(meeting_id)
    summary = db.get_summary(meeting_id)
    if not summary:
        return {"meeting_id": meeting_id, "summary_text": None, "key_points": [], "priorities": []}
    return {
        "meeting_id": meeting_id,
        "summary_text": summary["summary_text"],
        "key_points": summary["key_points"],
        "priorities": summary["priorities"],
    }


@app.get("/meetings/{meeting_id}/decisions")
def get_meeting_decisions(meeting_id: int):
    _require_meeting(meeting_id)
    return {"meeting_id": meeting_id, "decisions": db.get_decisions(meeting_id)}


@app.get("/meetings/{meeting_id}/action-items")
def get_meeting_action_items(meeting_id: int):
    _require_meeting(meeting_id)
    return {"meeting_id": meeting_id, "action_items": db.get_action_items(meeting_id)}


@app.get("/meetings/{meeting_id}/participants")
def get_meeting_participants(meeting_id: int):
    _require_meeting(meeting_id)
    return {"meeting_id": meeting_id, "participants": db.get_participants_for_meeting(meeting_id)}


@app.get("/meetings/{meeting_id}/deadlines")
def get_meeting_deadlines(meeting_id: int):
    _require_meeting(meeting_id)
    return {"meeting_id": meeting_id, "deadlines": db.get_deadlines_for_meeting(meeting_id)}


@app.get("/meetings/{meeting_id}/verify")
def verify_meeting(meeting_id: int):
    """Task 1 verification: confirms every piece of this meeting's data is
    retrievable and correctly linked (transcript/summary/decisions/action
    items/participants/deadlines), and flags cross-meeting contamination."""
    report = verify_meeting_retrieval(db, meeting_id)
    if not report.get("ok") and "checks" not in report:
        raise HTTPException(status_code=404, detail=report.get("reason", "Not found"))
    return report


# ===========================================================================
# Milestone 3 - Tasks 2 & 3: Embedding Generation + Vector Database
# ===========================================================================

@app.get("/meetings/{meeting_id}/vectors")
def get_meeting_vectors(meeting_id: int):
    """Task 3 meeting-to-vector mapping check: every vector stored for this
    meeting, each traceable back via its own meeting_id metadata."""
    _require_meeting(meeting_id)
    records = vector_store.get_by_meeting(meeting_id)
    return {
        "meeting_id": meeting_id,
        "vector_count": len(records),
        "vectors": [
            {"vector_id": r.id, "content_type": r.metadata.get("content_type"),
             "source_id": r.metadata.get("source_id"), "text": r.document[:200]}
            for r in records
        ],
    }


class EmbedRequest(BaseModel):
    meeting_id: int


@app.post("/embeddings/generate")
def generate_embeddings(request: EmbedRequest):
    """Tasks 2+3: (re)generate embeddings for one meeting and upsert them
    into the vector database. Safe to call repeatedly - existing vectors for
    this meeting are replaced, never duplicated (this is the 'Update
    embeddings' path)."""
    _require_meeting(request.meeting_id)
    try:
        chunks_indexed = index_meeting(db, request.meeting_id, embedding_service, vector_store)
    except Exception as e:  # noqa: BLE001
        logger.exception("Embedding generation failed for meeting_id=%s", request.meeting_id)
        raise HTTPException(status_code=502, detail="Embedding generation failed. See server logs for details.")
    return {"meeting_id": request.meeting_id, "chunks_indexed": chunks_indexed}


@app.post("/embeddings/sync")
def sync_embeddings():
    """Data synchronization: reindex every meeting currently in the
    database. Use after a bulk data change, or once when Milestone 3 is
    first deployed on top of an existing Milestone 2 database."""
    try:
        results = reindex_all_meetings(db, embedding_service, vector_store)
    except Exception as e:  # noqa: BLE001
        logger.exception("Embedding sync failed")
        raise HTTPException(status_code=502, detail="Embedding sync failed. See server logs for details.")
    return {"meetings_synced": len(results), "chunks_per_meeting": results}


@app.delete("/embeddings/{meeting_id}")
def delete_embeddings(meeting_id: int):
    """Task 3: remove every vector for a meeting - e.g. before deleting the
    meeting itself, so it can never remain searchable."""
    vector_store.delete_by_meeting(meeting_id)
    return {"meeting_id": meeting_id, "status": "deleted"}


# ===========================================================================
# Milestone 3 - Task 4: Semantic Search
# ===========================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    content_type: str = None
    meeting_id: int = None


@app.post("/search/meetings")
def search_meetings(request: SearchRequest):
    """Task 4: natural-language semantic search across historical meetings.
    e.g. {"query": "Which meeting discussed the database migration?"}"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    response = semantic_search(request.query, db=db, embedding_service=embedding_service,
                                vector_store=vector_store, top_k=request.top_k,
                                content_type_filter=request.content_type,
                                meeting_id_filter=request.meeting_id)
    return {
        "query": response.query,
        "results": [
            {
                "meeting_id": r.meeting_id, "meeting_title": r.meeting_title, "meeting_date": r.meeting_date,
                "content_type": r.content_type, "relevance_score": r.relevance_score, "matched_text": r.matched_text,
            }
            for r in response.results
        ],
        "search_completed_in_seconds": round(response.elapsed_seconds, 3),
        "within_latency_target": response.within_latency_target,
    }


# ===========================================================================
# Milestone 3 - Task 5: RAG Question Answering
# ===========================================================================

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask/meetings")
def ask_meetings(request: AskRequest):
    """Task 5: RAG question answering - grounded in retrieved meeting
    context, never unsupported/hardcoded. e.g.
    {"question": "What deadline was decided for the mobile application?"}"""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = answer_question(request.question, db=db, embedding_service=embedding_service,
                              vector_store=vector_store, top_k=request.top_k)
    return {
        "question": result.question,
        "answer": result.answer,
        "sources": [
            {"meeting_id": s.meeting_id, "meeting_title": s.filename,
             "content_type": s.content_type, "relevant_content": s.snippet}
            for s in result.sources
        ],
        "grounded": result.grounded,
        "used_local_fallback": result.used_local_fallback,
    }
