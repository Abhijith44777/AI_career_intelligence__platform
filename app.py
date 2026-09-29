"""
app.py
------
Streamlit Interface for:
    Milestone 1 - Audio Processing & Transcription
    Milestone 2 - Summarization & Action Extraction

Implements:
    M1 Task 1 - Whisper Transcription workflow
                Upload -> Process Audio -> Run Whisper -> Generate Transcript -> Display
    M1 Task 4 - Streamlit Interface
                File upload, Transcribe button, Processing status, Transcript display
    M2 Task 1/2 - LLM Processing -> Meeting Summary (via llm_service.py / summarization.py)
    M2 Task 3   - Action Item Extraction display (via action_items.py)
    M2 Task 4   - Participant & Responsibility Mapping display (via participants.py)
    M2 Task 5   - Persists everything to the database (via database.py)
    M2 Task 6   - This file + pipeline.py together ARE the "Processing API &
                  Service Integration" - the same stages api.py exposes over HTTP.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()  # picks up GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY from a local .env,
                # before any of the below read os.environ. Real environment
                # variables set via `set`/`export` still work and take
                # precedence if both are present.

from utils import (
    validate_file,
    process_audio,
    run_whisper,
    validate_transcript,
    save_transcript,
    transcript_matches_recording,
    get_audio_duration,
    ALLOWED_EXTENSIONS,
)
from llm_service import LLMService, LLMConfig, LLMServiceError
from summarization import summarize_meeting, render_summary_text
from action_items import extract_and_validate_action_items
from participants import map_participants_and_responsibilities
from database import Database

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "transcripts")
db = Database()

st.set_page_config(
    page_title="Career Intelligence Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', -apple-system, sans-serif; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    .app-header {
        background: linear-gradient(135deg, #3730A3 0%, #4F46E5 55%, #6366F1 100%);
        padding: 2.1rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.18);
    }
    .app-header h1 { margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.01em; }
    .app-header p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.95rem; font-weight: 400; }
    .app-header .pill {
        display: inline-block; background: rgba(255,255,255,0.16);
        padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.75rem;
        margin-top: 0.75rem; margin-right: 0.4rem; font-weight: 500;
    }

    .step-wrap { margin: 1.7rem 0 0.6rem 0; }
    .step-title {
        font-size: 1.08rem; font-weight: 700; color: #111827;
        display: flex; align-items: center; letter-spacing: -0.01em;
    }
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%;
        background: #4F46E5; color: white; font-weight: 700; font-size: 0.8rem;
        margin-right: 0.6rem; flex-shrink: 0;
    }
    .step-sub { color: #6B7280; font-size: 0.85rem; margin: 0.2rem 0 0 2.1rem; }

    div[data-testid="stMetricValue"] { color: #4338CA; font-weight: 700; }
    div[data-testid="stMetric"] {
        background: #F5F5FF; border: 1px solid #E5E7EB; border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }

    div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
    }

    .footer-note {
        color: #9CA3AF; font-size: 0.8rem; text-align: center; margin-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def step_header(number: str, title: str, subtitle: str = ""):
    """Consistent numbered section header used throughout the wizard."""
    st.markdown(
        f"""
        <div class="step-wrap">
            <div class="step-title"><span class="step-badge">{number}</span>{title}</div>
            {f'<div class="step-sub">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, icon: str, lines: list, empty_text: str = "None identified."):
    """A bordered card for one Structured Meeting Intelligence category."""
    with st.container(border=True):
        st.markdown(f"**{icon}&nbsp;&nbsp;{title}**")
        if lines:
            for line in lines:
                st.markdown(f"- {line}")
        else:
            st.caption(empty_text)


PRIORITY_COLORS = {
    "High": "background-color:#FEE2E2; color:#991B1B; font-weight:600;",
    "Medium": "background-color:#FEF3C7; color:#92400E; font-weight:600;",
    "Low": "background-color:#DBEAFE; color:#1E40AF; font-weight:600;",
}
STATUS_COLORS = {
    "Not Started": "background-color:#F3F4F6; color:#374151;",
    "In Progress": "background-color:#DBEAFE; color:#1E40AF;",
    "Done": "background-color:#DCFCE7; color:#166534;",
    "Completed": "background-color:#DCFCE7; color:#166534;",
}


def render_action_items_table(entries: list):
    """Renders action items as a styled, color-coded dataframe rather than
    a plain table - priority/status get a badge-style background color."""
    rows = [
        {
            "Task": e["action_item"].task,
            "Assignee": e["participant"].name,
            "Deadline": e["action_item"].deadline or "—",
            "Priority": e["action_item"].priority or "—",
            "Status": e["action_item"].status,
        }
        for e in entries
    ]
    df = pd.DataFrame(rows)
    try:
        # pandas >= 2.1
        styled = df.style.map(
            lambda v: PRIORITY_COLORS.get(v, ""), subset=["Priority"]
        ).map(
            lambda v: STATUS_COLORS.get(v, ""), subset=["Status"]
        )
    except AttributeError:
        # pandas < 2.1 - Styler.map didn't exist yet, use the older API
        styled = df.style.applymap(
            lambda v: PRIORITY_COLORS.get(v, ""), subset=["Priority"]
        ).applymap(
            lambda v: STATUS_COLORS.get(v, ""), subset=["Status"]
        )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🎙️ AI-Powered Career Intelligence Platform</h1>
        <p>Turn a raw meeting recording into a structured, actionable summary — transcription,
        LLM-driven analysis, and persistence in one pipeline.</p>
        <span class="pill">Milestone 1 · Transcription</span>
        <span class="pill">Milestone 2 · Summarization &amp; Action Extraction</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session state so results survive re-renders (e.g. clicking a download
# button doesn't wipe the screen).
if "transcript_result" not in st.session_state:
    st.session_state.transcript_result = None
if "processed_audio_path" not in st.session_state:
    st.session_state.processed_audio_path = None
if "source_filename" not in st.session_state:
    st.session_state.source_filename = None
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = None
if "meeting_id" not in st.session_state:
    st.session_state.meeting_id = None
if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = None
if "mapped_action_items" not in st.session_state:
    st.session_state.mapped_action_items = None

# ---------------------------------------------------------------------------
# Sidebar - model & provider settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### ⚙️ Transcription Settings")
    model_size = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Bigger = more accurate but slower. 'base' is a good default for meeting recordings.",
    )
    language = st.text_input(
        "Force language (optional)",
        value="",
        placeholder="e.g. en, hi, es — leave blank to auto-detect",
    )
    st.caption(f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    st.markdown("---")

    # -------------------------------------------------------------------
    # M2 Task 1 - LLM provider selection + live key status.
    # Shows exactly which provider will run and whether its key is
    # present *before* you click Generate, instead of only finding out
    # after a failed/degraded run.
    # -------------------------------------------------------------------
    st.markdown("#### 🧠 LLM Provider")
    PROVIDER_KEY_VARS = {
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "local": None,
    }
    provider_choice = st.selectbox(
        "Provider",
        list(PROVIDER_KEY_VARS.keys()),
        index=0,
        help="'gemini' is the default (free tier, no Anthropic key needed). "
             "'local' is a fully offline rule-based engine that never needs a key.",
    )
    key_var = PROVIDER_KEY_VARS[provider_choice]
    with st.container(border=True):
        if key_var is None:
            st.markdown("✅ **Offline engine**")
            st.caption("No API key required — always available.")
        elif os.environ.get(key_var):
            val = os.environ[key_var]
            masked = (val[:6] + "…" + val[-4:]) if len(val) > 12 else "•" * len(val)
            st.markdown(f"✅ **{key_var} detected**")
            st.caption(masked)
        else:
            st.markdown(f"⚠️ **{key_var} not set**")
            st.caption(
                "Will auto-fallback to the offline engine. Set it before "
                "launching Streamlit for real LLM output."
            )

# ---------------------------------------------------------------------------
# Step 1: Upload Meeting Recording  (Task 4 - File upload)
# ---------------------------------------------------------------------------
step_header("1", "Upload Meeting Recording")
uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # --- Task 2: File Upload Validation ---
    validation = validate_file(uploaded_file.name, uploaded_file.size)

    if not validation["valid"]:
        st.error(f"❌ Upload rejected: {validation['message']}")
        st.stop()
    else:
        st.success(f"✅ {validation['message']}")

    # ---------------------------------------------------------------------
    # Step 2: Transcribe button + Processing status  (Task 4)
    # ---------------------------------------------------------------------
    step_header("2", "Transcribe")
    transcribe_clicked = st.button("🚀 Transcribe", type="primary")

    if transcribe_clicked:
        st.session_state.source_filename = uploaded_file.name

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ---- Processing status shown at every stage ----
            status = st.status("Processing audio...", expanded=True)

            try:
                status.write("🔧 Extracting/normalizing audio (ffmpeg)...")
                processed_path = process_audio(raw_path, tmp_dir)

                duration = get_audio_duration(processed_path)
                status.write(f"⏱️ Detected duration: {duration:.1f} seconds")

                status.write(f"🧠 Loading Whisper model ('{model_size}')...")
                status.write("📝 Running transcription — this can take a while for long recordings...")

                result = run_whisper(
                    processed_path,
                    model_size=model_size,
                    language=language.strip() or None,
                )

                # ---- Task 3: Transcript Validation ----
                status.write("🔍 Validating transcript...")
                tv = validate_transcript(result["text"])
                if not tv["valid"]:
                    status.update(label="⚠️ Transcript validation warning", state="error")
                    st.warning(tv["message"])
                else:
                    match = transcript_matches_recording(result["text"], duration)
                    if not match["plausible"]:
                        st.warning(f"⚠️ {match['message']}")

                    saved_path = save_transcript(
                        result,
                        uploaded_file.name,
                        OUTPUT_DIR,
                    )
                    status.write(f"💾 Saved transcript to `{saved_path}`")

                    meeting_id = db.create_meeting(
                        filename=uploaded_file.name,
                        transcript=result["text"],
                        language=result.get("language"),
                        audio_duration_seconds=duration,
                    )
                    status.update(label="✅ Transcription complete!", state="complete")

                    st.session_state.transcript_result = result
                    st.session_state.audio_duration = duration
                    st.session_state.meeting_id = meeting_id
                    # New transcript => any previous meeting-intelligence result is stale
                    st.session_state.meeting_summary = None
                    st.session_state.mapped_action_items = None

            except RuntimeError as e:
                status.update(label="❌ Processing failed", state="error")
                st.error(str(e))
            except Exception as e:
                status.update(label="❌ Unexpected error", state="error")
                st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------------------------
# Step 3: Display Transcript  (Task 4)
# ---------------------------------------------------------------------------
if st.session_state.transcript_result:
    step_header("3", "Transcript")
    result = st.session_state.transcript_result

    tab_text, tab_segments = st.tabs(["📄 Full Text", "⏱️ Timestamped Segments"])

    with tab_text:
        st.text_area("Transcript", result["text"], height=260, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["text"],
            file_name=f"{os.path.splitext(st.session_state.source_filename)[0]}_transcript.txt",
        )

    with tab_segments:
        for seg in result["segments"]:
            st.write(f"**[{seg['start']:.1f}s - {seg['end']:.1f}s]** {seg['text']}")

    st.caption(f"Detected language: `{result['language']}`")

    # -----------------------------------------------------------------------
    # Milestone 2: LLM Processing -> Summary -> Action Extraction ->
    # Participant Mapping -> Database  (Tasks 1-6)
    # -----------------------------------------------------------------------
    step_header(
        "4", "Generate Meeting Intelligence",
        "Runs the Milestone 2 pipeline: LLM Processing → Summary → Action Extraction → Participant Mapping → Database",
    )

    generate_clicked = st.button("🧠 Generate Summary & Action Items", type="primary")

    if generate_clicked:
        m2_status = st.status("Running LLM processing...", expanded=True)
        try:
            m2_status.write(
                f"🧠 Calling LLM service (Task 1: prompt + structured output + validation) — "
                f"provider: `{provider_choice}`..."
            )
            meeting_summary = summarize_meeting(result["text"], LLMService(LLMConfig(provider=provider_choice)))

            m2_status.write("📝 Formatting meeting summary (Task 2)...")
            m2_status.write("✅ Extracting & validating action items (Task 3)...")
            validated_items = extract_and_validate_action_items(meeting_summary.action_items)

            m2_status.write("👥 Mapping participants & responsibilities (Task 4)...")
            mapping = map_participants_and_responsibilities(meeting_summary.participants, validated_items)

            m2_status.write("💾 Persisting summary & action items to the database (Task 5)...")
            db.save_summary(
                meeting_id=st.session_state.meeting_id,
                summary_text=meeting_summary.summary,
                key_points=meeting_summary.key_points,
                decisions=meeting_summary.decisions,
                priorities=meeting_summary.priorities,
            )
            for entry in mapping["action_items"]:
                item, participant = entry["action_item"], entry["participant"]
                participant_id = db.get_or_create_participant(participant.name)
                db.save_action_item(
                    meeting_id=st.session_state.meeting_id,
                    participant_id=participant_id,
                    task=item.task,
                    deadline=item.deadline,
                    priority=item.priority,
                    status=item.status,
                )
            for participant in mapping["participants"]:
                db.get_or_create_participant(participant.name)

            m2_status.update(label="✅ Meeting intelligence generated!", state="complete")
            st.session_state.meeting_summary = meeting_summary
            st.session_state.mapped_action_items = mapping["action_items"]

            if meeting_summary.summary.startswith("[Local rule-based engine"):
                st.info(
                    f"ℹ️ `{provider_choice}` had no valid API key available, so this result was "
                    "generated by the built-in offline rule-based engine instead. It still ran "
                    "the full Milestone 2 pipeline end-to-end — set the matching key (see sidebar "
                    "status) for higher-quality, genuinely semantic extraction."
                )

        except LLMServiceError as e:
            m2_status.update(label="❌ LLM processing failed", state="error")
            st.error(
                f"⚠️ {e}\n\nThis project does not require an Anthropic key. By default it uses "
                f"Google Gemini - set a `GEMINI_API_KEY` environment variable to enable it "
                f"(get one free at https://aistudio.google.com/apikey). If you skip this "
                f"entirely, the app will automatically fall back to a built-in offline engine "
                f"- this error means something else went wrong."
            )
        except Exception as e:
            m2_status.update(label="❌ Unexpected error", state="error")
            st.error(f"Something went wrong: {e}")

    # -- Display Milestone 2 results, if generated --------------------------
    if st.session_state.meeting_summary:
        ms = st.session_state.meeting_summary
        step_header(
            "5", "Structured Meeting Intelligence",
            "Task 1 output schema: Summary · Key Points · Decisions · Action Items · "
            "Participants · Deadlines · Priorities",
        )

        all_participant_names = sorted(
            {e["participant"].name for e in st.session_state.mapped_action_items}
            | set(ms.participants)
        ) if st.session_state.mapped_action_items else ms.participants

        # -- Quick-stats row --
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Participants", len(all_participant_names))
        m2.metric("Action Items", len(st.session_state.mapped_action_items or []))
        m3.metric("Decisions", len(ms.decisions))
        m4.metric("Deadlines", len(ms.deadlines))

        st.write("")
        col1, col2 = st.columns(2)

        with col1:
            info_card("Summary", "📋", [ms.summary] if ms.summary else [])
            info_card("Key Points", "🗝️", ms.key_points)
            info_card("Decisions", "✅", ms.decisions)
            info_card("Participants", "👥", all_participant_names)

        with col2:
            with st.container(border=True):
                st.markdown("**🧩&nbsp;&nbsp;Action Items**")
                if st.session_state.mapped_action_items:
                    render_action_items_table(st.session_state.mapped_action_items)
                else:
                    st.caption("No action items identified.")

            info_card("Deadlines", "⏰", ms.deadlines)
            info_card("Priorities", "🚦", ms.priorities)

        with st.expander("🧾 Task 2 formatted text (Summary / Key Decisions / Action Items)"):
            st.text_area("Formatted summary", render_summary_text(ms), height=260, label_visibility="collapsed")

        st.caption(f"Saved under meeting_id `{st.session_state.meeting_id}` in `meetings.db`")

# =============================================================================
# Milestone 3 - Meeting Knowledge Repository: Semantic Search & RAG Q&A
# =============================================================================
st.divider()
st.header("🔎 Historical Meeting Search")
st.caption(
    "Semantic search (Task 4) and RAG question answering (Task 5) over every "
    "meeting ever processed - not just the one above."
)

search_tab, ask_tab = st.tabs(["🔍 Semantic Search", "🤖 Ask Your Meetings"])

with search_tab:
    with st.form("semantic_search_form"):
        search_query = st.text_input(
            "Enter your question", placeholder="e.g. Which meeting discussed the database migration?"
        )
        search_submitted = st.form_submit_button("Search", type="primary")

    if search_submitted and search_query.strip():
        from embedding_service import EmbeddingService
        from vector_store import VectorStore
        from semantic_search import semantic_search, LATENCY_TARGET_SECONDS

        with st.spinner("Searching meeting history..."):
            response = semantic_search(search_query, db=db, embedding_service=EmbeddingService(),
                                        vector_store=VectorStore())

        latency_label = f"{response.elapsed_seconds:.2f}s"
        if response.within_latency_target:
            st.caption(f"✅ Search completed in: {latency_label}  (target: under {LATENCY_TARGET_SECONDS:.0f}s)")
        else:
            st.caption(f"⚠️ Search completed in: {latency_label}  (target: under {LATENCY_TARGET_SECONDS:.0f}s)")

        if not response.results:
            st.info("No matching meetings found yet. Process a few meetings above first.")
        for r in response.results:
            with st.container(border=True):
                st.markdown(f"**{r.meeting_title}**")
                st.caption(f"Date: {r.meeting_date}  ·  Relevant content type: {r.content_type}  ·  "
                           f"Similarity: `{r.relevance_score:.3f}`")
                st.write(f"Relevant content: \u201c{r.matched_text}\u201d")

with ask_tab:
    st.markdown("**What would you like to know about previous meetings?**")
    with st.form("rag_ask_form"):
        question = st.text_input(
            "Question", placeholder="e.g. What deadline was decided for the mobile application?",
            label_visibility="collapsed",
        )
        ask_submitted = st.form_submit_button("Ask", type="primary")

    if ask_submitted and question.strip():
        from embedding_service import EmbeddingService
        from vector_store import VectorStore
        from rag_qa import answer_question

        with st.spinner("Retrieving context and generating a grounded answer..."):
            rag_result = answer_question(question, db=db, embedding_service=EmbeddingService(),
                                          vector_store=VectorStore(), llm_service=LLMService())

        st.markdown("### Answer")
        st.write(rag_result.answer)
        if rag_result.used_local_fallback:
            st.caption("ℹ️ Answered with the local extractive fallback (no LLM key configured for Q&A).")
        if rag_result.sources:
            st.markdown("### Sources")
            for s in rag_result.sources:
                with st.container(border=True):
                    st.markdown(f"**Meeting:** {s.filename}  ·  **Meeting ID:** {s.meeting_id}")
                    st.caption(f"Relevant content ({s.content_type}): {s.snippet}")

st.markdown(
    """
    <div class="footer-note">
        Milestone 1 · Audio Processing &amp; Transcription &nbsp;+&nbsp;
        Milestone 2 · Summarization &amp; Action Extraction &nbsp;+&nbsp;
        Milestone 3 · Knowledge Repository, Semantic Search &amp; RAG Q&amp;A &nbsp;·&nbsp;
        AI-Powered Career Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)
