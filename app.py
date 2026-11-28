"""Streamlit chat UI for the Parliament Bills chatbot.

Run from the project directory: streamlit run app.py
Groq is optional. With no key, or with Local mode selected, the app uses
local planning, the metadata database and local document retrieval/composition.
"""
from __future__ import annotations

import os
from typing import Any

import streamlit as st

from step3_build_brain import Brain, MODEL_NAME as EMBED_MODEL
from step4_talk import FactsMode, answer_question, format_sources


def brain_model(brain_dir: str) -> str:
    """Return the embedding model recorded for this brain, if available."""
    path = os.path.join(brain_dir, "cache_model.txt")
    try:
        with open(path, encoding="utf-8") as handle:
            recorded = handle.read().strip()
        return recorded or EMBED_MODEL
    except (OSError, UnicodeError):
        return EMBED_MODEL


st.set_page_config(page_title="Parliament Bills Chatbot", page_icon="🏛️", layout="wide")

# Visual theme: quiet, document-like, institutional. Palette lives in
# .streamlit/config.toml; this block only shapes the chat itself - hides the
# stock chrome, gives each message a card, and keeps the header type calm.
st.markdown(
    """
    <style>
      #MainMenu, footer { visibility: hidden; }
      .block-container { padding-top: 2.2rem; max-width: 1080px; }
      h1 { font-weight: 700; letter-spacing: 0.2px; margin-bottom: 0; }
      [data-testid="stChatMessage"] {
          background: #ffffff;
          border: 1px solid #e4ddce;
          border-radius: 10px;
          padding: 0.35rem 0.9rem;
          margin-bottom: 0.55rem;
          box-shadow: 0 1px 2px rgba(30, 42, 58, 0.05);
      }
      [data-testid="stChatMessage"] p { line-height: 1.55; }
      .stButton > button {
          background: #ffffff; border: 1px solid #cdc4b2; border-radius: 8px;
      }
      .stButton > button:hover { border-color: #7b1e26; color: #7b1e26; }
      [data-testid="stChatInput"] textarea { border-radius: 10px; }
      [data-testid="stSidebar"] { border-right: 1px solid #e3ddce; }
      /* stock Streamlit chrome (Deploy button) is not part of this app */
      .stDeployButton, [data-testid="stDeployButton"],
      [data-testid="stAppDeployButton"], .stAppDeployButton { display: none !important; }
      /* Ctrl+P prints the conversation itself, not the controls */
      @media print {
          [data-testid="stSidebar"], [data-testid="stChatInput"],
          [data-testid="stHeader"], .stButton { display: none !important; }
          .block-container { max-width: 100%; padding-top: 0; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# The index folder and embedding model are NOT user settings: they must match
# the index built by step3_build_brain.py, so they are fixed in code instead of
# being offered as fields anyone can type into.
brain_dir = "./brain"
top_n = 3

# Settings live behind a gear button, as in any app - the page itself stays clean.
# (Older Streamlit versions without popover fall back to the sidebar.)
_head_left, _head_right = st.columns([11, 1])
with _head_right:
    st.write("")
    _settings = st.popover("⚙️", help="Settings") if hasattr(st, "popover") else st.sidebar
with _settings:
        document_search = st.checkbox("Enable document search (clause-level answers)", value=False,
            help="Off: answers from the records table and the introduced-bill PDFs. On: also searches the indexed clause text.")
        mode = st.radio(
            "Answer mode", ["exact", "explain"], index=0,
            help="Exact quotes the source wording. Explain rephrases it in simpler language when a language model is connected.",
        )
        show_chunks = st.checkbox("Show source extracts with answers", value=False,
            help="Displays the exact passages each answer is based on.")
        local_only = st.checkbox("Local mode (no internet calls)", value=True,
            help="Every step stays on this computer.")
        st.caption("Print / save as PDF: press **Ctrl+P**.")
with _head_left:
    st.title("Parliament Bills Chatbot")

st.caption(
    "Lok Sabha and Rajya Sabha bills, 2004-2026. Answers come from the local "
    "metadata table and the indexed bill documents."
)
# Freshness, said out loud. The table is a snapshot, not a live feed: a bill
# introduced after the day it was fetched cannot be in it, so the app names
# that day instead of implying "everything up to today".
try:
    from step4_talk import _as_of
    st.caption(f"Metadata as of **{_as_of()}** - bills introduced after that day are not in this copy.")
except Exception:
    pass

suggested_model = brain_model(brain_dir) if document_search else EMBED_MODEL
embed_model = suggested_model

# No on-screen key field: an external language model is used only when a key
# is set in the environment; otherwise every answer is extractive and local.
api_key = os.environ.get("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Loading bill search index ...")
def load_brain(path: str, model: str) -> Brain:
    return Brain(path, model)


brain = FactsMode()
if document_search:
    try:
        brain = load_brain(brain_dir, embed_model.strip() or suggested_model)
    except (SystemExit, Exception) as exc:
        st.warning('Document search is unavailable; metadata and intro-PDF questions still work. '
                   f'Index message: {type(exc).__name__}: {exc}')
if isinstance(brain, FactsMode):
    st.caption('Answering from: records table + introduced-bill PDFs.')
else:
    st.caption('Answering from: records table + introduced-bill PDFs + clause index.')

# UI history is kept separately from rendered Markdown so the planner receives
# clean question/answer pairs rather than source-formatting text.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None

examples = [
    "Who proposed Rajya Sabha Bill No. II of 2023?",
    "Who proposed Lok Sabha Bill No. 370 of 2019?",
    "When was Lok Sabha Bill No. 370 of 2019 passed?",
]
columns = st.columns(len(examples))
for column, example in zip(columns, examples):
    if column.button(example, use_container_width=True):
        st.session_state.pending_question = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about a bill, its provisions, dates or status ...")
prompt = prompt or st.session_state.pop("pending_question", None)

if prompt:
    prompt = str(prompt).strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # An empty key deliberately selects the local path. Checking Local mode
        # also suppresses Groq even when GROQ_API_KEY is set in the environment.
        effective_key = api_key.strip()
        no_llm = bool(local_only or not effective_key)
        with st.spinner("Checking bill metadata and searching its text ..."):
            try:
                brain.last_filters = st.session_state.last_filters
                reply, hits, used = answer_question(
                    brain,
                    prompt,
                    mode=mode,
                    api_key=effective_key,
                    llm_models=None,
                    top_n=top_n,
                    history=st.session_state.conversation_history[-4:],
                    last_filters=st.session_state.last_filters,
                    no_llm=no_llm,
                    mock_llm=False,
                )
            except SystemExit as exc:
                reply, hits, used = str(exc) or "The answer service failed.", [], "error"
            except Exception as exc:
                reply, hits, used = f"{type(exc).__name__}: {exc}", [], "error"

        st.session_state.last_filters = getattr(brain, "last_filters", None)
        full = f"{reply}\n\n**Answer path:** `{used}`"
        if hits:
            full += "\n\n**Sources**\n```text\n" + format_sources(hits) + "\n```"
        if show_chunks and hits:
            for number, hit in enumerate(hits, 1):
                text = hit.get("chunk", {}).get("text", "")
                full += f"\n\n**Extract {number}:**\n\n{text[:1200]}"

        st.session_state.messages.append({"role": "assistant", "content": full})
        st.session_state.conversation_history.append((prompt, str(reply)))
        st.session_state.conversation_history = st.session_state.conversation_history[-4:]
        with st.chat_message("assistant"):
            st.markdown(full)

st.divider()
st.caption(
    "Coverage ends 10 August 2026 (the day the records were downloaded). "
    "Bills whose documents were published as scans are answered from their "
    "records only; the app says so when that happens."
)
