# User guide

## Start

```powershell
Set-Location -LiteralPath "D:\PRJ1(Parliament)\bills-chatbot"
.\.venv\Scripts\python.exe .\start_trial.py
```

The browser opens the chat. Alternatives: `streamlit run app.py` (same UI), or
`python server.py` for the lightweight interface at `http://127.0.0.1:8000`.

## Asking

- Name a bill by **House + number + year**: `Who introduced Rajya Sabha Bill
  No. II of 2023?` - the three-part identity is the strongest form.
- Or by **title**: `When was the Citizenship (Amendment) Bill passed?`
- Follow-ups work: after a bill is named, `when was it passed?`, `who
  introduced it?`, `what are its clauses?` stay on that bill. A follow-up
  never overrides a question that names a new year, house or number.
- Counts: `How many bills were introduced in 2024?` - the answer states the
  counting rule it used.

## Settings (the ⚙️ button, top right)

- **Enable document search** - off: metadata + introduced PDFs only. On: also
  searches the clause index (99,997 chunks). Clause questions need this on.
- **Answer mode** - *exact* favours source wording; *explain* rephrases in
  simpler language when a language model is connected. With none connected the
  words are identical and the answer path says so.
- **Local mode** - on by default; no API calls at all.

## What it will not do

- Answer about bills outside 2004-2026, or bills that do not exist - it
  refuses and says why.
- Settle a source conflict silently - if the CSV and a signed PDF name
  different people, both are shown and flagged as a conflict.
- Describe provisions of a bill whose text was never readable (580 bills,
  mostly scanned PDFs) - it gives the dates and says the text is unavailable.

## If something looks wrong

Run `start_trial.py --check`. If that passes but an answer still looks wrong,
note the exact question and the "Answer path" line shown under the answer -
those two lines diagnose almost everything.
