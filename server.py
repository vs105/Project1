#!/usr/bin/env python3
"""
STEP 4 (part 4) - WEB UI (Flask backend for the bills chatbot)
================================================================
Same brain, same answer function as the command line bot. Zero new logic.
The page in ui/index.html only talks to this server.

Run   : python server.py                     (then open http://localhost:8000)
Offline demo: python server.py --mock        (no key, no charges possible)

Endpoints
  GET  /              -> the chat page
  GET  /api/health    -> brain stats + whether an LLM key is present
  POST /api/ask       -> question, mode, top_n, chunks_only, mock_llm, sources
  GET  /api/examples  -> demo questions
"""
import argparse
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

from step3_build_brain import Brain, MODEL_NAME as EMBED_MODEL
from step4_talk import (answer_question, format_sources, LLM_MODELS, REFUSAL)

# The last few questions and answers, so "what punishment does it prescribe"
# can inherit the bill from the question before it. Safe as a plain list
# because app.run is called with threaded=False, so one request is handled at
# a time and there is no race on this list.
HISTORY = []

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")

app = Flask(__name__, static_folder=None)

# The brain is loaded ONCE and reused for every question (it is big).
BRAIN = {"brain": None, "model": None, "error": None, "dir": "",
         "rerank": True}


def load_brain(brain_dir, model, rerank=True):
    """Loads the brain once. Keeps the error so /api/health can explain it."""
    if BRAIN["brain"] is not None:
        return BRAIN["brain"]
    try:
        # The re-ranker is on by default, because quoting the wrong clause is
        # worse than waiting a second. Pass --no-rerank to turn it off.
        BRAIN["rerank"] = rerank
        BRAIN["brain"] = Brain(brain_dir, model, rerank=rerank)
        BRAIN["model"] = model
    except SystemExit as e:
        BRAIN["error"] = str(e) or "the brain could not be loaded"
    except Exception as e:  # never let a bad brain kill the server
        BRAIN["error"] = f"{type(e).__name__}: {e}"
    return BRAIN["brain"]


@app.after_request
def allow_preview(resp):
    """The demo is shown from a proxied host, so let it call this API."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.get("/api/health")
def health():
    brain = BRAIN["brain"]
    prs = 0
    if brain is not None:
        prs = sum(1 for c in brain.chunks.values()
                  if c.get("source", "sansad") != "sansad")
    return jsonify({
        "ok": brain is not None,
        "error": BRAIN["error"],
        "brain_dir": BRAIN["dir"],
        "model": BRAIN["model"],
        "chunks": len(brain.chunks) if brain else 0,
        "prs_chunks": prs,
        "groq_models": LLM_MODELS,
        "has_key": bool(os.environ.get("GROQ_API_KEY")
                        or os.environ.get("GEMINI_API_KEY")),
        "facts_bills": _facts_count(),
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
        return jsonify({"error": BRAIN["error"] or "the brain is not loaded"}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "type a question first"}), 400

    mode = data.get("mode") or "exact"
    if mode not in ("exact", "explain"):
        mode = "exact"
    try:
        top_n = max(1, min(5, int(data.get("top_n") or 3)))
    except (TypeError, ValueError):
        top_n = 3
    chunks_only = bool(data.get("chunks_only"))
    mock_llm = bool(data.get("mock_llm"))
    sources = data.get("sources") or "all"
    api_key = (data.get("api_key") or os.environ.get("GROQ_API_KEY", "")
               or os.environ.get("GEMINI_API_KEY", ""))

    last = getattr(brain, "last_filters", (None, None))

    # Chat memory: the client may send the last turns, otherwise the server
    # uses what it remembers. Either way the planner gets the previous
    # question, which is how "it" and "that" get resolved to a real bill.
    history = data.get("history") or HISTORY[-4:]

    try:
        reply, hits, used = answer_question(brain, question, mode, api_key,
                                            None, top_n, history, last,
                                            chunks_only, mock_llm)
    except SystemExit as e:
        # quota / key / network problems arrive here: show them, never crash.
        return jsonify({"error": str(e) or "the answer service failed"}), 200
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 200

    HISTORY.append((question, reply))
    while len(HISTORY) > 6:
        HISTORY.pop(0)

    year_now, no_now = getattr(brain, "last_filters", (None, None))
    src = []
    for n, h in enumerate(hits, 1):
        c = h["chunk"]
        src.append({
            "n": n,
            "score": h.get("score"),
            "bill_title": c.get("bill_title", "?"),
            "bill_no": c.get("bill_no", "?"),
            "year": c.get("year", "?"),
            "ministry": c.get("ministry", "?"),
            "status": c.get("status", "?"),
            "slot": c.get("slot", "?"),
            "clause": (c.get("label") or ("clause " + c["clause_no"]
                       if c.get("clause_no") else c.get("part", ""))),
            "part": c.get("part", ""),
            "clause_no": c.get("clause_no", ""),
            "source_file": c.get("source_file", "?"),
            "language": c.get("language", "?"),
            "tag": "official" if c.get("source", "sansad") == "sansad" else "PRS summary",
            "text": c.get("text", ""),
        })
    return jsonify({
        "reply": reply,
        "sources": src,
        "used": used,
        "refusal": REFUSAL,
        "filters": {"year": year_now, "bill_no": no_now},
        "sources_text": format_sources(hits) if hits else "",
    })


def _facts_count():
    """How many real bills the facts layer can answer from (no key needed)."""
    try:
        from step4_talk import get_facts
        db = get_facts()
        return len(db.bills) if db else 0
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--brain", default="./brain")
    ap.add_argument("--model", default="")
    ap.add_argument("--mock", action="store_true",
                    help="demo mode: the page asks for mock_llm, so no key is used")
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the cross-encoder (faster, but a worse clause "
                         "can be chosen)")
    a = ap.parse_args()
    no_rerank = a.no_rerank

    model = a.model
    if not model:  # always match the model the brain was actually built with
        p = os.path.join(a.brain, "cache_model.txt")
        model = (open(p, encoding="utf-8").read().strip()
                 if os.path.exists(p) else MODEL_NAME)

    BRAIN["dir"] = a.brain
    print(f"loading brain from {a.brain} (model {model}) ...", flush=True)
    brain = load_brain(a.brain, model, not no_rerank)
    if brain is None:
        print(f"BRAIN FAILED: {BRAIN['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"brain ready: {len(brain.chunks)} chunks, "
          f"facts layer {_facts_count()} bills", flush=True)
    if a.mock:
        print("mock mode: no API key is used, no charges are possible", flush=True)
    print(f"open http://localhost:{a.port}", flush=True)
    # threaded=False on purpose: the facts layer opens ONE sqlite connection and
    # sqlite refuses to use it from another thread ("SQLite objects created in a
    # thread can only be used in that same thread"). One request at a time is
    # fine for a demo and keeps every answer exact. If you ever need parallel
    # requests, give each thread its own connection instead.
    app.run(host=a.host, port=a.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
