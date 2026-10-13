"""
STEP 4 (part 3) - APP (Streamlit chat UI for the bills bot)
============================================================
The pretty face on Step 4: same Brain + same answer function, zero new logic.
If step4_talk.py works, this works - it literally calls answer_question().

Install:  pip install streamlit
Run   :  streamlit run app.py   (opens http://localhost:8501 in your browser)
"""
import os

import streamlit as st

from step3_build_brain import Brain, MODEL_NAME as EMBED_MODEL
from step4_talk import answer_question, format_sources, try_facts


def brain_model(brain_dir):
    """Which embedding model built THIS brain? (Step 3 records it on disk.)
    Auto-matching means you can never pick the wrong model by accident."""
    p = os.path.join(brain_dir, "cache_model.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    return EMBED_MODEL


st.set_page_config(page_title="Lok Sabha Bills Chatbot", page_icon="🏛️", layout="wide")
st.title("🏛️ Lok Sabha Bills Chatbot (2004–2026)")
st.caption("Exact mode quotes official Sansad wording · Explain mode simplifies it · "
           "Every fact is cited.")

# ----------------------------------------------------------------------------
# SIDEBAR - all settings, no code changes needed.
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    brain_dir = st.text_input("Brain folder (Step 3 output)", "./brain")
    embed_model = st.text_input("Embedding model (auto-matched to brain)",
                                brain_model(brain_dir))
    mode = st.radio("Answer mode", ["exact", "explain"], index=0,
                    help="exact = official wording · explain = simple words")
    top_n = st.slider("Extracts per answer", 1, 5, 3)
    chunks_only = st.checkbox("Chunks only (no LLM, no key needed)")
    show_chunks = st.checkbox("Show retrieved extracts (debug)")
    api_key = st.text_input("Gemini API key", os.environ.get("GEMINI_API_KEY", ""),
                            type="password", help="Free: https://aistudio.google.com/apikey")


# ----------------------------------------------------------------------------
# BRAIN - loaded once, reused for every question.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading brain ...")
def load_brain(brain_dir, embed_model):
    return Brain(brain_dir, embed_model)

try:
    brain = load_brain(brain_dir, embed_model)
except SystemExit as e:  # Brain explains itself plainly; show that text.
    st.error(str(e) or "Brain failed to load.")
    need = brain_model(brain_dir)
    st.warning(f'This brain was built with `{need}` - '
               'make sure "Embedding model" above shows the same name.')
    st.stop()

# ----------------------------------------------------------------------------
# CHAT - history in session_state, so it survives every click.
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

examples = ["What is the Finance Bill 2024 about?",
            "Tell me about Bill No. 370 of 2019.",
            "वित्त विधेयक 2024 किस बारे में है?"]
cols = st.columns(3)
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.pending = ex

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about any Lok Sabha Bill (2004–2026) ...")
prompt = prompt or st.session_state.pop("pending", None)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    facts_reply = try_facts(brain, prompt, st.session_state.get("last_filters"))
    if facts_reply:  # dates / status / counts: exact answer from the database, no key needed.
        reply, hits, used = facts_reply, [], "sansad-metadata"
    elif not chunks_only and not api_key and not os.environ.get("GROQ_API_KEY"):
        reply = ("⚠️ Add your free Gemini API key in the sidebar "
                 "(https://aistudio.google.com/apikey), set a free Groq key "
                 "(GROQ_API_KEY), or tick **Chunks only**.")
        hits, used = [], "none"
    else:
        with st.spinner("Reading official extracts ..."):
            try:  # quota/key/net errors arrive as SystemExit: show, don't crash.
                reply, hits, used = answer_question(brain, prompt, mode, api_key,
                                                    None, top_n, "all",
                                                    chunks_only, False,
                                                    st.session_state.get("last_filters"))
            except SystemExit as e:
                reply = str(e) or "The answer service failed - retry, or tick Chunks only."
                hits, used = [], "none"
    st.session_state.last_filters = getattr(brain, "last_filters", (None, None))
    full = reply + (f"\n\n**Sources** (via {used}):\n```\n{format_sources(hits)}\n```"
                    if hits else "")
    if show_chunks and hits:
        for n, h in enumerate(hits, 1):
            full += f"\n\n<details><summary>extract [{n}]</summary>\n\n{h['chunk']['text'][:1200]}\n\n</details>"
    with st.chat_message("assistant"):
        st.markdown(full)
    st.session_state.messages.append({"role": "assistant", "content": full})

st.divider()
st.caption("Data: official Sansad PDFs (truth) + PRS summaries (translator only). "
           "On any conflict, Sansad wins.")
