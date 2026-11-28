#!/usr/bin/env python3
"""Flask API and small browser UI for the Parliament Bills chatbot.

Run locally:  python server.py
Open:         http://127.0.0.1:8000
Offline:      python server.py --mock

The server binds to loopback by default. Use --host 0.0.0.0 only when you
intentionally want other machines to reach it; this demo has no login system.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from step3_build_brain import Brain, MODEL_NAME as EMBED_MODEL
from step4_talk import FactsMode, answer_question, format_sources, LLM_MODELS, REFUSAL

HISTORY: list[tuple[str, str]] = []
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
app = Flask(__name__, static_folder=None)

BRAIN: dict[str, Any] = {
    "brain": None,
    "model": None,
    "error": None,
    "dir": "",
    "rerank": False,
}
SERVER_NO_LLM = False
MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 4


def load_brain(brain_dir: str, model: str, rerank: bool = False):
    """Load the large search brain once and retain a readable failure reason."""
    if BRAIN["brain"] is not None:
        return BRAIN["brain"]
    BRAIN["dir"] = brain_dir
    BRAIN["model"] = model
    BRAIN["rerank"] = bool(rerank)
    BRAIN["error"] = None
    try:
        BRAIN["brain"] = Brain(brain_dir, model, rerank=rerank)
    except SystemExit as exc:
        BRAIN["error"] = str(exc) or "The brain could not be loaded."
    except Exception as exc:
        BRAIN["error"] = f"{type(exc).__name__}: {exc}"
    return BRAIN["brain"]


def _last_filters(brain):
    """Return the full (year, bill number, house) identity tuple, or None."""
    filters = getattr(brain, "last_filters", None)
    if not isinstance(filters, (tuple, list)) or not filters:
        return None
    values = list(filters[:3])
    while len(values) < 3:
        values.append(None)
    return tuple(values)


def _facts_count() -> int:
    """How many bill records are currently available to the metadata layer."""
    try:
        from step4_talk import get_facts
        db = get_facts()
        return len(db.bills) if db is not None else 0
    except Exception:
        return 0


def _client_filters(raw):
    """Validate a browser's previous (year, bill number, house) context.

    The client keeps this per browser session. This avoids using another
    browser's most recently discussed bill for a follow-up such as "what did
    it provide?". Invalid or stale values are ignored rather than trusted.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    year_raw, number_raw = raw[0], raw[1]
    if not isinstance(year_raw, (str, int)) or isinstance(year_raw, bool):
        return None
    if not isinstance(number_raw, (str, int)) or isinstance(number_raw, bool):
        return None
    year = str(year_raw).strip()
    number = str(number_raw).strip().upper()
    if not re.fullmatch(r"20\d{2}", year) or not 2004 <= int(year) <= 2026:
        return None
    if not re.fullmatch(r"(?:\d{1,4}|[IVXLCDM]{1,8})", number):
        return None
    house = raw[2] if len(raw) > 2 else None
    if house not in ("Lok Sabha", "Rajya Sabha"):
        house = None
    return year, number, house


def _clean_history(raw) -> list[tuple[str, str]]:
    """Accept only a small list of plain question/answer pairs from the client."""
    if not isinstance(raw, list):
        return []
    cleaned: list[tuple[str, str]] = []
    for item in raw[-MAX_HISTORY_TURNS:]:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        question, answer = item
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        question = question.strip()[:MAX_QUESTION_CHARS]
        answer = answer.strip()[:4000]
        if question and answer:
            cleaned.append((question, answer))
    return cleaned


def _request_api_key(data) -> str:
    """Resolve a request key without defeating an explicit local-mode choice.

    An omitted field may use GROQ_API_KEY from the server environment. An
    explicitly supplied blank field means the browser intentionally selected
    local mode, so it must override the environment key.
    """
    if "api_key" in data:
        value = data.get("api_key")
        return value.strip() if isinstance(value, str) else ""
    return os.environ.get("GROQ_API_KEY", "").strip()


def _metadata_source_label(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "sansad":
        return "Sansad metadata"
    if value == "rti":
        return "RTI Wiki metadata"
    return "Provenance unknown"


@app.get("/")
def index():
    index_path = os.path.join(UI_DIR, "index.html")
    if not os.path.isfile(index_path):
        return (
            "<h2>Chat page is missing</h2>"
            "<p>The API is available, but <code>ui/index.html</code> was not found. "
            "Restore the included <code>ui</code> folder.</p>",
            404,
        )
    return send_from_directory(UI_DIR, "index.html")


@app.get("/api/health")
def health():
    brain = BRAIN["brain"]
    chunks = brain.chunks.values() if brain is not None else []
    rti_chunks = sum(
        1 for chunk in chunks
        if str(chunk.get("source_primary", "")).strip().lower() == "rti"
    )
    return jsonify({
        "ok": brain is not None,
        "error": BRAIN["error"],
        "brain_dir": BRAIN["dir"],
        "model": BRAIN["model"],
        "chunks": len(brain.chunks) if brain is not None else 0,
        "rti_primary_chunks": rti_chunks,
        "groq_models": LLM_MODELS,
        "has_key": bool(os.environ.get("GROQ_API_KEY")),
        "facts_bills": _facts_count(),
        "local_mode": SERVER_NO_LLM,
    })


@app.get("/api/examples")
def examples():
    return jsonify({"examples": [
        "hi",
        "What is the Finance Bill 2024 about?",
        "Give me all the details of the Finance Bill 2024",
        "Which communities are covered under the Citizenship Amendment Act?",
        "When was the Citizenship (Amendment) Bill 2019 passed?",
        "How many bills were assented in 2019?",
        "What punishment does the Triple Talaq Act 2019 prescribe?",
    ]})


@app.post("/api/ask")
def ask():
    brain = BRAIN["brain"]
    if brain is None:
        return jsonify({"error": BRAIN["error"] or "The brain is not loaded."}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with a question field."}), 400

    question = data.get("question")
    if not isinstance(question, str):
        return jsonify({"error": "The question must be text."}), 400
    question = question.strip()
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"Question is too long (maximum {MAX_QUESTION_CHARS} characters)."}), 413

    mode = data.get("mode", "exact")
    if mode not in ("exact", "explain"):
        mode = "exact"
    try:
        top_n = max(1, min(5, int(data.get("top_n", 3))))
    except (TypeError, ValueError):
        top_n = 3

    chunks_only = data.get("chunks_only") is True
    mock_llm = data.get("mock_llm") is True
    api_key = _request_api_key(data)

    history = _clean_history(data.get("history"))
    # New browser clients send their own context. Older clients retain the
    # original single-user fallback, which is suitable for localhost demos.
    previous_filters = (
        _client_filters(data.get("last_filters"))
        if "last_filters" in data else None
    )

    # Keep the compatibility mirror aligned with this browser's context
    # before the synchronous request. This server is intentionally single-
    # threaded, so another request cannot overwrite it mid-answer.
    brain.last_filters = previous_filters

    try:
        reply, hits, used = answer_question(
            brain,
            question,
            mode=mode,
            api_key=api_key,
            llm_models=None,
            top_n=top_n,
            history=history,
            last_filters=previous_filters,
            no_llm=bool(SERVER_NO_LLM or chunks_only),
            mock_llm=mock_llm,
        )
    except SystemExit as exc:
        return jsonify({"error": str(exc) or "The answer service failed."}), 503
    except Exception as exc:
        app.logger.exception("Question answering failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


    sources = []
    for number, hit in enumerate(hits, 1):
        chunk = hit.get("chunk") or {}
        bill_no = str(chunk.get("bill_no", "?"))
        clause_no = chunk.get("clause_no")
        clause_label = chunk.get("label") or (
            f"clause {clause_no}" if clause_no else chunk.get("part", "")
        )
        source_primary = chunk.get("source_primary", "")
        sources.append({
            "n": number,
            "score": hit.get("score"),
            "bill_title": chunk.get("bill_title", "?"),
            "bill_no": bill_no,
            "year": chunk.get("year", "?"),
            "house": chunk.get("house", ""),
            "ministry": chunk.get("ministry", "?"),
            "status": chunk.get("status", "?"),
            "slot": chunk.get("slot", "?"),
            "chunk_id": chunk.get("chunk_id", ""),
            "clause": clause_label,
            "part": chunk.get("part", ""),
            "clause_no": clause_no or "",
            "source_file": chunk.get("source_file", "?"),
            "language": chunk.get("language", "?"),
            "metadata_source": _metadata_source_label(source_primary),
            "source_primary": source_primary or "unknown",
            "text": chunk.get("text", ""),
        })

    filters = _last_filters(brain)
    return jsonify({
        "reply": reply,
        "sources": sources,
        "used": used,
        # Only a real refusal, never a canned message shown beside a good answer.
        "refusal": REFUSAL if not reply else "",
        "filters": {
            "year": filters[0] if filters else None,
            "bill_no": filters[1] if filters else None,
            "house": filters[2] if filters else None,
        },
        "sources_text": format_sources(hits) if hits else "",
    })


def main() -> None:
    global SERVER_NO_LLM
    parser = argparse.ArgumentParser(description="Run the Parliament Bills chatbot web server.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="default 127.0.0.1; use 0.0.0.0 only for intentional network access")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--brain", default="./brain")
    parser.add_argument("--facts-only", action="store_true", help="metadata + local intro PDFs; do not open Chroma")
    parser.add_argument("--model", default="")
    parser.add_argument("--mock", action="store_true",
                        help="force local mode for every request; no Groq calls")
    parser.add_argument("--no-rerank", action="store_true",
                        help="skip the cross-encoder (faster, but may reduce clause precision)")
    args = parser.parse_args()

    SERVER_NO_LLM = bool(args.mock)
    model = args.model.strip()
    if not model:
        model_path = os.path.join(args.brain, "cache_model.txt")
        if os.path.isfile(model_path):
            with open(model_path, encoding="utf-8") as handle:
                model = handle.read().strip()
        if not model:
            model = EMBED_MODEL

    print(f"Loading brain from {args.brain} (model {model}) ...", flush=True)
    if args.facts_only:
        brain = FactsMode()
        BRAIN.update(brain=brain, error=None, dir=None, model='facts-only')
        SERVER_NO_LLM = True
    else:
        brain = load_brain(args.brain, model, rerank=not args.no_rerank)
    if brain is None:
        print(f"BRAIN FAILED: {BRAIN['error']}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"Brain ready: {len(brain.chunks)} chunks; facts layer {_facts_count()} bills",
        flush=True,
    )
    if SERVER_NO_LLM:
        print("Local mode enabled: Groq calls are disabled for every request.", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "WARNING: server is exposed beyond this computer. This demo has no authentication; "
            "use only on a trusted network.",
            file=sys.stderr,
            flush=True,
        )
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    # The metadata layer currently keeps a SQLite connection for the process;
    # single-threaded serving avoids cross-thread SQLite errors in this demo.
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
