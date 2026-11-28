# Parliament Bills Chatbot

A local chatbot over Indian Parliament bills (Lok Sabha and Rajya Sabha,
2004-2026). It answers from a metadata table of 5,169 bills and an index of
the bill documents themselves - entirely offline. An optional external
language-model key enables a simpler-worded "explain" mode; without it every
answer is extractive, with the source named.

Data snapshot: **2026-08-10**. Bills introduced after that day are not in this
copy, and the app says so rather than guessing.

## Quick start (Windows)

```powershell
Set-Location -LiteralPath "D:\PRJ1(Parliament)\bills-chatbot"
.\.venv\Scripts\python.exe .\start_trial.py
```

A browser opens with the chat. Tick **Enable document search** to read clause
text; leave it off for metadata-only answers (dates, status, introducer).

## Check it works

```powershell
.\.venv\Scripts\python.exe .\start_trial.py --check   # unit + contract + live questions
.\.venv\Scripts\python.exe .\qa_suite.py              # integration / system / adversarial
.\.venv\Scripts\python.exe .\bill_audit.py            # every bill, 4 questions each
```

See **docs/TESTING.md** for what each level proves and the latest results.

## Project layout

| path | what it is |
|---|---|
| `app.py` | Streamlit chat interface |
| `server.py`, `ui/` | lightweight Flask chat interface (alternative) |
| `start_trial.py` | one-command launcher and `--check` gate |
| `bills_db.py` | metadata table: identity resolution, dates, counts, conflicts |
| `step4_talk.py` | the answering layer: planning, refusal, exact/explain |
| `attribution.py` | "who proposed" evidence, with source conflicts flagged |
| `download_bills.py`, `step2_organize_clean.py`, `step3_build_brain.py` | the data pipeline (download -> clean -> index) |
| `bills_data/`, `clean_data/`, `brain/` | metadata CSV, cleaned text, search index |
| `qa_suite.py`, `trial_selftest.py`, `web_ui_contract_selftest.py`, `bills_db_selftest.py`, `bill_audit.py` | the test battery |
| `make_report.py` | audits the whole folder and writes a dated report |
| `docs/` | USER-GUIDE, TESTING, DATA |

## Honest limits

- 580 of the 5,169 bills have no readable text (scanned PDFs or no document
  published); for those, only dates/status/introducer are answered, and the
  app says so. Row-level list: run `bill_audit.py` and see `has_text` in
  `BILL-AUDIT-ALL.csv`.
- 1,163 bills have no individual introducer recorded by Sansad; the ministry
  is given and the blank is stated.
- Source conflicts (e.g. CSV vs signed PDF naming different people) are
  reported as conflicts, never silently settled.
