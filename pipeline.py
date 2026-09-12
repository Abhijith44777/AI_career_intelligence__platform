"""
pipeline.py
------------
Milestone 2 - Task 6: Processing API & Service Integration

Wires Milestone 1 and Milestone 2 together into the single end-to-end
flow from the spec:

    Upload Meeting
         |
      Whisper
         |
     Transcript
         |
    LLM Processing
         |
      Summary
         |
   Action Extraction
         |
  Participant Mapping
         |
       Database

Every stage's failure is caught and reported with which stage failed,
rather than one bare traceback - both the Streamlit UI (app.py) and the
API (api.py) call this same function so the two interfaces can never
drift apart in behavior.
"""

from dataclasses import dataclass, field

# Milestone 1
from utils import (
    validate_file,
    process_audio,
    run_whisper,
    validate_transcript,
    save_transcript,
    get_audio_duration,
)

# Milestone 2
from llm_service import LLMService, LLMServiceError
from summarization import summarize_meeting, MeetingSummary
from action_items import extract_and_validate_action_items
from participants import map_participants_and_responsibilities
from database import Database


class PipelineError(Exception):
    """Raised with a `.stage` attribute identifying which pipeline stage failed,
    so callers can show a precise error instead of a generic failure."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


@dataclass
class PipelineResult:
    meeting_id: int
    transcript: str
    language: str
    audio_duration_seconds: float
    summary: MeetingSummary
    action_items: list = field(default_factory=list)   # list of {"action_item":.., "participant":..}
    participants: list = field(default_factory=list)


def process_meeting_recording(
    raw_file_path: str,
    original_filename: str,
    file_size_bytes: int,
    db: Database = None,
    llm_service: LLMService = None,
    whisper_model_size: str = "base",
) -> PipelineResult:
    """
    Full Milestone 1 + Milestone 2 pipeline, matching the Task 6 flow
    diagram exactly. Each stage below is labeled with the stage name it
    corresponds to in that diagram.
    """
    db = db or Database()
    llm_service = llm_service or LLMService()

    # --- Upload Meeting: validate the file (Milestone 1, Task 2) -----------
    validation = validate_file(original_filename, file_size_bytes)
    if not validation["valid"]:
        raise PipelineError("Upload Validation", validation["message"])

    # --- Whisper: process audio + transcribe (Milestone 1, Task 1) ---------
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_audio_path = process_audio(raw_file_path, tmp_dir)
            duration = get_audio_duration(processed_audio_path)
            whisper_result = run_whisper(processed_audio_path, model_size=whisper_model_size)
    except RuntimeError as e:
        raise PipelineError("Whisper", str(e))

    transcript_text = whisper_result["text"]

    # --- Transcript: validate before spending LLM calls on bad input -------
    transcript_check = validate_transcript(transcript_text)
    if not transcript_check["valid"]:
        raise PipelineError("Transcript Validation", transcript_check["message"])

    save_transcript(whisper_result, original_filename, output_dir="transcripts")

    # Persist the meeting/transcript row now so every later stage has a
    # meeting_id to attach to, even if a later stage fails.
    meeting_id = db.create_meeting(
        filename=original_filename,
        transcript=transcript_text,
        language=whisper_result.get("language"),
        audio_duration_seconds=duration,
    )

    # --- LLM Processing -> Summary: structured extraction (Milestone 2, Task 1+2) ---
    try:
        meeting_summary = summarize_meeting(transcript_text, llm_service)
    except LLMServiceError as e:
        raise PipelineError("LLM Processing", str(e))

    db.save_summary(
        meeting_id=meeting_id,
        summary_text=meeting_summary.summary,
        key_points=meeting_summary.key_points,
        decisions=meeting_summary.decisions,
        priorities=meeting_summary.priorities,
    )

    # --- Action Extraction (Milestone 2, Task 3) ----------------------------
    validated_action_items = extract_and_validate_action_items(meeting_summary.action_items)

    # --- Participant Mapping (Milestone 2, Task 4) ---------------------------
    mapping = map_participants_and_responsibilities(meeting_summary.participants, validated_action_items)

    # --- Database: persist participants + linked action items (Task 5) -------
    for entry in mapping["action_items"]:
        item = entry["action_item"]
        participant = entry["participant"]
        participant_id = db.get_or_create_participant(participant.name)
        db.save_action_item(
            meeting_id=meeting_id,
            participant_id=participant_id,
            task=item.task,
            deadline=item.deadline,
            priority=item.priority,
            status=item.status,
        )

    # Also persist participants who were mentioned but have no action item,
    # so they still show up as meeting attendees.
    for participant in mapping["participants"]:
        db.get_or_create_participant(participant.name)

    return PipelineResult(
        meeting_id=meeting_id,
        transcript=transcript_text,
        language=whisper_result.get("language"),
        audio_duration_seconds=duration,
        summary=meeting_summary,
        action_items=mapping["action_items"],
        participants=mapping["participants"],
    )
