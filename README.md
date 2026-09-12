# AI-Powered Career Intelligence Platform
## Milestone 1 — Audio Processing & Transcription

This milestone builds the audio-to-text pipeline that the rest of the
platform (resume/interview intelligence, skill extraction, etc.) will sit
on top of. It uses **OpenAI Whisper** for speech-to-text and **Streamlit**
for the interface.

---

## 📁 Project Structure

```
career_intelligence_platform/
├── app.py                # Task 1 + Task 4 — Streamlit interface & workflow
├── utils.py               # Task 1, 2, 3 — core logic (validation, processing, Whisper, saving)
├── accuracy_test.py       # Task 5 — accuracy testing (WER-based)
├── test_validation.py     # Automated tests for Task 2 & Task 3
├── requirements.txt
├── transcripts/           # auto-created — saved transcripts (.json + .txt)
└── sample_data/
    ├── audio/              # put test recordings here for accuracy_test.py
    └── ground_truth/       # matching human-verified transcripts (.txt)
```

---

## ⚙️ Setup

```bash
# 1. Install ffmpeg (required by Whisper to read audio/video)
#    macOS:   brew install ffmpeg
#    Ubuntu:  sudo apt install ffmpeg
#    Windows: choco install ffmpeg

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## Task 1 — Whisper Transcription (workflow)

**Flow:** Upload Meeting Recording → Process Audio → Run Whisper →
Generate Transcript → Display Transcript

| Step | Where | What happens |
|---|---|---|
| Upload Meeting Recording | `app.py` → `st.file_uploader` | User selects an audio/video file |
| Process Audio | `utils.process_audio()` | ffmpeg extracts audio (if video) and normalizes to 16kHz mono WAV |
| Run Whisper | `utils.run_whisper()` | Loads a cached Whisper model and transcribes the WAV |
| Generate Transcript | `utils.run_whisper()` return value | Returns full text + timestamped segments + detected language |
| Display Transcript | `app.py` | Shown in a text area + timestamped-segments tab, with a download button |

**Why audio is normalized first:** Whisper decodes to 16kHz mono internally
regardless of input — doing this explicitly with ffmpeg means processing
failures (corrupted files, unreadable codecs) surface as a clear error
*before* the expensive model call, not a confusing crash mid-transcription.

**Model size trade-off** (selectable in the sidebar):
`tiny` → `base` → `small` → `medium` → `large` (accuracy increases, speed decreases).

---

## Task 2 — File Upload Validation

Implemented in `utils.validate_file()`, called immediately after upload in
`app.py`.

Checks performed:
- ✅ **Format allow-list** — audio: `.mp3 .wav .m4a .flac .ogg .aac .wma`,
  video: `.mp4 .mov .mkv .avi .webm`
- ✅ **Empty file rejection** — 0-byte / near-empty uploads
- ✅ **Size cap** — rejects files over 500MB (configurable) to avoid
  hanging the app
- ✅ **Human-readable error messages** — e.g. *"Unsupported file type
  '.pdf'. Supported formats are: .aac, .avi, .flac..."*

Verify it yourself:
```bash
python test_validation.py
```

---

## Task 3 — Transcript Validation

Implemented in `utils.validate_transcript()`,
`utils.transcript_matches_recording()`, and `utils.save_transcript()`.

- **Not empty** — rejects blank/whitespace-only output
- **Matches the recording** — a words-per-minute plausibility check against
  the actual audio duration (catches silent truncation, e.g. only the
  first 30 seconds of a 20-minute meeting got transcribed)
- **Degenerate-output detection** — flags Whisper's known failure mode of
  looping on one word on noisy/silent audio
- **Saved correctly** — every transcript is written to `transcripts/` as
  both a `.json` (full data: text + timestamped segments + language +
  source filename + generation time) and a plain `.txt` sidecar

---

## Task 4 — Streamlit Interface

`app.py` provides:
- **File upload** — drag-and-drop or browse, restricted to supported types
- **Transcribe button** — explicit trigger rather than auto-running on
  upload, so large files aren't processed by accident
- **Processing status** — a live `st.status()` block showing each pipeline
  stage (extracting audio → loading model → transcribing → validating →
  saving) so long-running jobs don't look frozen
- **Transcript display** — full text view + a timestamped-segments view,
  with a one-click `.txt` download

---

## Task 5 — Accuracy Testing

`accuracy_test.py` computes **Word Error Rate (WER)** between Whisper's
output and a human-verified ground-truth transcript for each test
recording, using the `jiwer` library (with a built-in Levenshtein
fallback if `jiwer` isn't installed).

**Accuracy = (1 − WER) × 100**, broken down into:
- **Substitutions** — wrong word used
- **Insertions** — extra word Whisper added that wasn't spoken
- **Deletions** — word that was spoken but missing from the transcript

### How to run it
1. Add test recordings to `sample_data/audio/` (e.g. `meeting1.wav`)
2. Add the matching *actual* transcript to
   `sample_data/ground_truth/meeting1.txt`
3. Run:
   ```bash
   python accuracy_test.py --model base
   ```
4. It prints a per-file breakdown and an overall average, and reports
   **PASS/FAIL against the ≥90% accuracy target** from the milestone spec.

### If accuracy is below 90%
- Try a larger model (`--model small` or `--model medium`)
- Check the recording for background noise / overlapping speakers
  (Whisper accuracy drops significantly with cross-talk)
- Double-check the ground-truth `.txt` file is actually correct —
  a bad reference transcript will look like a Whisper error

---

## Suggested Test Plan (bringing all 5 tasks together)

1. Run `python test_validation.py` → confirms Task 2 & 3 logic (12/12 pass)
2. Launch `streamlit run app.py`, upload a short real recording, verify
   the full Task 1 flow renders a transcript with no errors
3. Try uploading a `.pdf` or a 0-byte file to confirm Task 2's rejection
   messages appear correctly
4. Check `transcripts/` after a run to confirm Task 3's save step worked
5. Gather 3-5 real meeting clips with known ground truth and run
   `accuracy_test.py` to confirm Task 5's ≥90% target

---

# Milestone 2 — Summarization & Action Extraction

Builds on Milestone 1's transcripts: takes the raw transcript text and turns
it into structured meeting intelligence — a summary, decisions, action
items, participants, deadlines and priorities — validated and saved to a
database.

## Additional Project Structure

```
career_intelligence_platform/
├── llm_service.py          # Task 1 — LLM service, prompts, schema validation, chunking, retries
├── summarization.py        # Task 2 — Meeting summarization module
├── action_items.py         # Task 3 — Action item extraction & normalization
├── participants.py         # Task 4 — Participant & responsibility mapping
├── database.py             # Task 5 — SQLite data model & persistence
├── pipeline.py             # Task 6 — End-to-end orchestration (M1 + M2)
├── api.py                  # Task 6 — FastAPI service exposing the pipeline
├── test_llm_service.py     # Tests for Task 1
├── test_action_items.py    # Tests for Task 3
├── test_participants.py    # Tests for Task 4
├── test_database.py        # Tests for Task 5
├── local_extraction.py     # Zero-key rule-based engine (default fallback)
├── test_local_extraction.py # Tests for local_extraction.py
└── meetings.db              # auto-created SQLite database
```

## Setup — additional step (optional — no key required)

**No API key is required.** If you run the app or API with no key configured
at all, `LLMService` automatically uses a built-in, fully offline rule-based
engine (`local_extraction.py`) to run the complete Milestone 2 pipeline —
summary, decisions, action items, participants, deadlines, priorities. This
also kicks in automatically as a fallback any time a real provider's key is
missing or invalid, so a bad key degrades the *output quality*, not the app.

For higher-quality, genuinely semantic extraction, set an LLM API key. The
default provider is **Google Gemini** (no Anthropic key needed, free tier
available):

```bash
# Windows (Command Prompt)
set GEMINI_API_KEY=your-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-key-here"

# macOS / Linux
export GEMINI_API_KEY=your-key-here
```

Get a free key at https://aistudio.google.com/apikey.

To use Anthropic or OpenAI instead, set `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` and construct `LLMService(LLMConfig(provider="anthropic"))`
or `LLMService(LLMConfig(provider="openai", model="gpt-4o"))`. To force the
offline engine even if a key is present (e.g. for a fast, free demo), use
`LLMService(LLMConfig(provider="local"))`.

---

## Task 1 — LLM Service & Prompt Engineering

`llm_service.py` is the whole LLM layer, deliberately kept UI/DB-agnostic:

- **Approved LLM configuration** — `LLMConfig` dataclass: provider, model,
  temperature, token limits, retry settings, all in one place
- **Reusable prompt templates** — `EXTRACTION_PROMPT_TEMPLATE` (single-pass)
  and `MERGE_PROMPT_TEMPLATE` (for combining chunked results)
- **Structured output handling** — every response must match
  `MEETING_INTELLIGENCE_SCHEMA` (summary, key_points, decisions,
  action_items, participants, deadlines, priorities); `parse_json_response()`
  tolerates markdown-fenced or prose-wrapped JSON
- **Input/output validation** — `validate_structured_output()` checks every
  field's presence and type, including per-item validation of each action
  item's sub-schema
- **Token/context-limit handling** — `chunk_transcript()` splits long
  transcripts on sentence boundaries under a configurable character budget;
  each chunk is processed separately, then merged via `MERGE_PROMPT_TEMPLATE`
  (falling back to a deterministic `naive_merge_partials()` if the merge
  call itself fails)
- **Retry & failure handling** — `_call_llm_raw()` retries transient errors
  with exponential backoff; `_call_and_validate()` retries the *whole
  generation* if the JSON fails schema validation (a fresh generation often
  self-corrects formatting mistakes). If every retry against the configured
  provider still fails (missing key, invalid key, network/auth error),
  `extract_meeting_intelligence()` transparently falls back to
  `local_extraction.py` per chunk rather than raising — see
  `local_extraction.py` for the zero-key rule-based engine used both as the
  default provider and as this fallback.

## Task 2 — Meeting Summarization Module

`summarization.py` wraps the LLM service into a stable `MeetingSummary`
object and renders it in the exact human-readable format from the spec:

```
Summary:
The team discussed the mobile application launch and assigned
API integration and UI testing responsibilities.

Key Decisions:
- Continue with the planned mobile application launch.

Action Items:
- Complete API integration - Ravi - Friday
- Prepare UI testing report - Priya
```

## Task 3 — Action Item Extraction Engine

`action_items.py` hardens the LLM's raw action items into a guaranteed
consistent shape:
- **Assigned participant** — defaults to `"Unassigned"` if missing
- **Deadline** — defaults to `"No deadline"` if missing
- **Priority** — normalized to High/Medium/Low; inferred from urgency
  keywords ("urgent", "ASAP", "no rush", etc.) when the LLM's value is
  missing or invalid
- **Status** — normalized to Not Started / In Progress / Completed,
  tolerating common LLM phrasing variants ("done", "ongoing", "todo")

Items with no task text are dropped rather than saved as noise.

## Task 4 — Participant & Responsibility Mapping

`participants.py` provides a `ParticipantRegistry` used per-meeting:
- **Correct identification** — names come from both the LLM's
  `participants` list and every action item's `assignee` field
- **Consistent mapping** — `normalize_name()` handles case/whitespace
  variants ("ravi kumar" / "RAVI KUMAR" → "Ravi Kumar"); a fuzzy-match
  fallback (`difflib`, 85% similarity threshold) catches minor spelling
  variance without merging unrelated people
- **Multiple participants** — no limit on how many distinct people are
  tracked per meeting
- **Unknown participants handled safely** — blank/"unassigned"/"n/a"
  assignees all resolve to one shared `"Unknown"` record rather than
  crashing or fabricating a name
- **No duplicate records** — the registry checks for an existing match
  before creating a new one; the database layer (Task 5) enforces this
  again with a `UNIQUE` constraint across meetings
- **Responsibilities linked correctly** — `map_participants_and_responsibilities()`
  returns each action item paired with its resolved `Participant`, ready
  to persist with the right foreign key

## Task 5 — Meeting Data Model & Database Persistence

`database.py` implements the flow: **Transcript → AI Processing →
Structured Output → Validation → Database**, using SQLite (stdlib
`sqlite3`, zero extra setup).

Schema:
- `meetings` — filename, transcript, language, audio duration, timestamp
- `participants` — `name` with a `UNIQUE` constraint (the hard guarantee
  against duplicate records across the whole database, not just per-meeting)
- `summaries` — summary text + JSON-encoded key points/decisions/priorities,
  linked to a meeting
- `action_items` — task/deadline/priority/status, linked to both a meeting
  and a participant via foreign keys

`Database.get_meeting_full(meeting_id)` returns everything about one
meeting (transcript + summary + action items) in a single call — the shape
both the Streamlit UI and the API return.

## Task 6 — Processing API & Service Integration

The full pipeline from the spec:

```
Upload Meeting → Whisper → Transcript → LLM Processing → Summary →
Action Extraction → Participant Mapping → Database
```

is implemented once in **`pipeline.py`** (`process_meeting_recording()`) and
used by both interfaces, so they can never drift apart in behavior:

- **`app.py`** (Streamlit) — Milestone 1's UI now has a 4th and 5th step:
  a "Generate Summary & Action Items" button that runs the Milestone 2
  pipeline on the transcript and displays the summary, action items table,
  and participant list
- **`api.py`** (FastAPI) — the same pipeline exposed over HTTP:
  - `POST /process-meeting` — upload a file, run the full pipeline, get
    back structured JSON
  - `GET /meetings` — list all processed meetings
  - `GET /meetings/{id}` — full detail for one meeting
  - `GET /health` — liveness check

  Run with:
  ```bash
  uvicorn api:app --reload
  ```
  Then open http://127.0.0.1:8000/docs for interactive API documentation.

Each pipeline stage's failure is caught and tagged with a `stage` name
(`PipelineError.stage`), so both the UI and the API can report exactly
where processing failed instead of a generic error.

---

## Running the Milestone 2 Test Suite

All logic that doesn't require a live LLM call is covered by automated
tests (59 tests total across both milestones, all passing), including the
zero-key local rule-based engine, which is fully offline and fully tested:

```bash
python test_llm_service.py       # Task 1 — schema validation, JSON parsing, chunking, merging
python test_action_items.py      # Task 3 — normalization, priority inference, validation
python test_participants.py      # Task 4 — name normalization, dedup, unknown handling
python test_database.py          # Task 5 — full persistence round-trips
python test_local_extraction.py  # Zero-key engine — sentence scoring, decisions, action items, deadlines
```

The one thing these tests intentionally don't cover is a live call to a
real LLM provider (Gemini/Anthropic/OpenAI inside `LLMService._call_llm_raw`)
— that needs a real API key and network access, so it's exercised by
actually running the app end-to-end rather than in automated tests. The
*fallback path* that engages when that call is unavailable, however, is
fully covered — that's exactly what `test_local_extraction.py` tests.

## Suggested End-to-End Test Plan

1. Run `streamlit run app.py` — **no API key needed.** Upload a short
   meeting recording, click **Transcribe** — confirms Milestone 1 still
   works
2. Click **Generate Summary & Action Items** with no key set — confirms the
   zero-key local engine runs the full Task 1–4 pipeline end-to-end (you'll
   see an info banner noting the offline engine was used)
3. (Optional, for higher-quality output) Set `GEMINI_API_KEY` and repeat
   step 2 — confirms the real LLM path (Task 1: LLM call + validation,
   Task 2: summary formatting, Task 3: action items, Task 4: participant
   mapping) all in one pass
4. Check the **Summary**, **Action Items**, and **Participants** tabs render
   correctly
5. Open `meetings.db` with any SQLite browser (or `sqlite3 meetings.db`) to
   confirm Task 5's persistence
6. Run `uvicorn api:app --reload` and `POST` the same file to
   `/process-meeting` via the `/docs` page to confirm Task 6's API layer
   returns the same result
