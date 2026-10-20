# Frontend Integration Guide

## Backend Overview

- **Framework:** Flask (Python)
- **Entry file:** `server.py`
- **Default port:** 8000
- **Auth:** None (no login, no tokens)
- **CORS:** Already enabled via `@app.after_request` — allows all origins, `Content-Type` header, `GET`/`POST`/`OPTIONS` methods
- **Streaming:** None — all responses are standard JSON

## API Endpoints

| Method | Path            | Request body                                                                                   | Response shape                                                                                                               | Auth | Streaming |
|--------|-----------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------|-----------|
| GET    | `/`             | —                                                                                              | HTML (serves `ui/index.html`)                                                                                                | No   | No        |
| GET    | `/api/health`   | —                                                                                              | `{ ok, error, brain_dir, model, chunks, prs_chunks, groq_models, has_key, facts_bills }`                                    | No   | No        |
| GET    | `/api/examples` | —                                                                                              | `{ examples: string[] }`                                                                                                     | No   | No        |
| POST   | `/api/ask`      | `{ question, mode?, top_n?, chunks_only?, mock_llm?, sources?, api_key?, history?: [q,a][] }` | `{ reply, sources: Source[], used, refusal, filters: {year, bill_no}, sources_text }` or `{ error }` on 400/503             | No   | No        |

### Source object shape (in `/api/ask` response)

```json
{
  "n": 1,
  "score": 0.85,
  "bill_title": "The Finance Bill, 2024",
  "bill_no": "3",
  "year": "2024",
  "ministry": "Ministry of Finance",
  "status": "billPassedInBothHousesFile",
  "slot": "...",
  "clause": "Preamble",
  "part": "",
  "clause_no": "",
  "source_file": "filename.pdf",
  "language": "english",
  "tag": "official",
  "text": "full chunk text..."
}
```

## Message Flow

1. **App boot:** `GET /api/health` to confirm the brain is loaded; `GET /api/examples` to fetch example questions for the empty state.
2. **User sends a message:** `POST /api/ask` with `{ question }` (and optionally `mode`, `top_n`, `history`).
3. **Server responds** with `{ reply, sources, used, ... }`. Response is not streamed; the full answer arrives in one JSON body.
4. **History:** The server maintains its own in-memory history (last 4 turns). The client may also send `history` as an array of `[question, answer]` tuples to override it.

## What the backend does NOT provide

- **Chat sessions / conversation list:** No endpoint to create, list, rename, or delete conversations. The server keeps a single flat in-memory history.
- **User authentication / profiles:** None.
- **File uploads:** Not supported.
- **Streaming responses:** Not supported (full JSON response only).
- **Chat title generation:** Not provided; must be derived client-side from the first user message.
