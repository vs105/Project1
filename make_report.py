#!/usr/bin/env python3
"""
make_report.py - collect everything a reviewer needs, as a few SMALL text files.

Why this exists
---------------
The project's data is large (bills_data ~1.8 GB, clean_data ~234 MB, brain
several GB) and cannot be uploaded anywhere. But almost none of that size is
information: it is PDFs and binary vectors, already summarised inside
kept_chunks.json and the metadata CSV. This script reads the project in place,
never modifies anything, and writes a few text files that are small enough to
send (normally well under 2 MB) while still being complete enough to answer:

  * what is installed, and which versions
  * which files exist, their sizes, and whether the code files are unmodified
  * the exact shape of the metadata table (rows, columns, blanks, duplicates)
  * the exact shape of the brain (chunks, bills, years, parts, slots)
  * WHICH BILLS HAVE NO INDEXED TEXT AT ALL - the real cause of "the bot
    cannot answer about that bill"
  * how many PDFs exist per stage, and how many file names the parser cannot
    read (those are silently skipped by step 2)
  * a handful of real chunks, verbatim, so answers can be traced

Usage (from the project folder, the one containing bills_data\\ and brain\\):

    .\\.venv\\Scripts\\python.exe make_report.py

    # or point it at the folder explicitly, and choose an output folder
    .\\.venv\\Scripts\\python.exe make_report.py --root "D:\\PRJ1(Parliament)\\bills-chatbot"
    .\\.venv\\Scripts\\python.exe make_report.py --out "C:\\Users\\me\\Desktop\\report"

It writes a folder `_report_<timestamp>` plus `report-<timestamp>.zip` beside it.
Send the ZIP. Nothing else in the project is read-write-touched: the only writes
are inside the output folder.

Options
-------
  --root PATH     project folder to inspect (default: found by walking up from cwd)
  --out PATH      where to write the report (default: <root>/_report_<stamp>)
  --no-zip        leave the folder, do not zip it
  --samples N     how many brain chunks to quote verbatim (default 30)
  --max-mb N      refuse to write a report bigger than this (default 8)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import random
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------
# What the report collects. Kept in one place so nothing is forgotten silently.
# --------------------------------------------------------------------------
PACKAGES = ["streamlit", "chromadb", "sentence-transformers", "torch", "numpy",
            "pandas", "rank-bm25", "pymupdf", "PyMuPDF", "groq", "pydantic",
            "flask", "onnxruntime", "huggingface-hub", "transformers"]

# Directories whose CONTENTS are never walked: they are huge and binary, and
# their size/count is reported from the folder level instead of per file.
SKIP_DIRS = {".venv", "venv", "env", "__pycache__", ".git", "node_modules",
             "chroma_db", "pdfs", "_report", "v2_trial_old", "test_data"}

HASH_LIMIT = 25 * 1024 * 1024      # do not hash files above 25 MB
READ_LIMIT = 25 * 1024 * 1024      # do not parse JSON/CSV above 25 MB in memory

# A downloaded file is named  <index>_<billno>_<year>_<name>.pdf  (see step 2).
PDF_NAME = re.compile(r"^[^_]+_([0-9A-Za-z]+)_(\d{4})_")
YEAR_IN_TITLE = re.compile(r"\b(19|20)\d{2}\b")


class Report:
    """Collects sections and the list of problems found along the way."""

    def __init__(self, root: Path, out: Path):
        self.root = root
        self.out = out
        self.problems: list[str] = []
        self.files: list[dict] = []
        self.started = _dt.datetime.now()

    # ---- small helpers ---------------------------------------------------
    def note(self, text: str) -> None:
        self.problems.append(text)

    def write(self, name: str, text: str) -> Path:
        path = self.out / name
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def human(self, n: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"


# --------------------------------------------------------------------------
# 1. Environment
# --------------------------------------------------------------------------
def section_environment(r: Report) -> str:
    lines = ["# 1. Environment", "",
             f"- date: `{r.started:%Y-%m-%d %H:%M:%S}`",
             f"- python: `{sys.version.split()[0]}` at `{sys.executable}`",
             f"- in a virtualenv: `{sys.prefix != getattr(sys, 'base_prefix', sys.prefix)}`",
             f"- executable exists: `{os.path.exists(sys.executable)}`",
             f"- platform: `{platform.platform()}`",
             f"- machine: `{platform.machine()}`",
             f"- cwd: `{os.getcwd()}`",
             f"- project root: `{r.root}`",
             "", "## Installed packages", "", "| package | version |", "|---|---|"]
    try:
        from importlib import metadata
        seen = {}
        for name in PACKAGES:
            try:
                seen[name] = metadata.version(name)
            except Exception:
                seen[name] = "not installed"
        for name, ver in seen.items():
            lines.append(f"| {name} | {ver} |")
    except Exception as exc:                                    # pragma: no cover
        lines.append(f"| (could not list packages) | {type(exc).__name__}: {exc} |")
        r.note(f"package listing failed: {type(exc).__name__}: {exc}")
    lines += ["", "## Environment variables that change behaviour", ""]
    for key in ("BILLS_CSV", "GROQ_API_KEY", "PYTHONUTF8", "HF_HOME",
                "SENTENCE_TRANSFORMERS_HOME", "TOKENIZERS_PARALLELISM"):
        val = os.environ.get(key)
        if key == "GROQ_API_KEY":
            lines.append(f"- `{key}`: {'set (hidden)' if val else 'not set'}")
        else:
            lines.append(f"- `{key}`: `{val}`" if val else f"- `{key}`: not set")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 2. Files present (the answer to "what is in these folders?")
# --------------------------------------------------------------------------
def sha256_of(path: Path, limit: int = HASH_LIMIT) -> str:
    size = path.stat().st_size
    if size > limit:
        return "(not hashed: larger than %d MB)" % (limit // (1024 * 1024))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_files(r: Report) -> None:
    """Walk the project, recording size/mtime/hash for everything except the
    deliberately skipped trees."""
    for base, dirs, files in os.walk(r.root):
        base_p = Path(base)
        dirs[:] = sorted(d for d in dirs
                         if d not in SKIP_DIRS and not d.startswith("_report"))
        for name in sorted(files):
            p = base_p / name
            try:
                st = p.stat()
            except OSError:
                continue
            r.files.append({"path": r.rel(p), "size": st.st_size,
                            "mtime": _dt.datetime.fromtimestamp(st.st_mtime)
                                     .strftime("%Y-%m-%d %H:%M")})


def section_files(r: Report) -> str:
    """Two levels only: a folder-level map (so huge trees stay readable), then
    the code files in full detail."""
    by_top: dict[str, dict] = {}
    for f in r.files:
        top = f["path"].split(os.sep)[0] if os.sep in f["path"] else "(root)"
        agg = by_top.setdefault(top, {"files": 0, "bytes": 0})
        agg["files"] += 1
        agg["bytes"] += f["size"]

    # One extra level for the data folders, which are the interesting ones.
    for sub in ("brain", "clean_data", os.path.join("bills_data")):
        agg = {"files": 0, "bytes": 0}
        found = False
        p = r.root / sub
        if p.is_dir():
            for entry in p.rglob("*"):
                if entry.is_file():
                    agg["files"] += 1
                    found = True
                    try:
                        agg["bytes"] += entry.stat().st_size
                    except OSError:
                        pass
        if found:
            by_top[sub + " (incl. subfolders)"] = agg

    lines = ["# 2. Files", "",
             "Folder totals. Big trees (`brain/`, `bills_data/pdfs/`, `.venv/`) are summed,",
             "never listed file by file.", "",
             "| folder | files | size |", "|---|---:|---:|"]
    for top in sorted(by_top, key=lambda k: -by_top[k]["bytes"]):
        a = by_top[top]
        lines.append(f"| `{top}` | {a['files']:,} | {r.human(a['bytes'])} |")

    lines += ["", "## Code and document files (with hashes)", "",
              "Hashes let you tell whether a file was edited after this report was made.",
              "", "| file | size | modified | sha256 (first 16) |", "|---|---:|---|---|"]
    code_ext = (".py", ".html", ".md", ".txt", ".json", ".csv", ".toml", ".cfg",
                ".ini", ".yml", ".yaml", ".bat", ".ps1")
    interesting = [f for f in r.files
                   if f["path"].lower().endswith(code_ext)
                   and not f["path"].startswith("brain" + os.sep)
                   and "chroma_db" not in f["path"]]
    for f in sorted(interesting, key=lambda x: x["path"]):
        if f["size"] > 8 * 1024 * 1024:
            continue
        digest = sha256_of(r.root / f["path"])
        short = digest[:16] if not digest.startswith("(") else digest
        lines.append(f"| `{f['path']}` | {r.human(f['size'])} | {f['mtime']} | `{short}` |")
    if not interesting:
        lines.append("| (none found) | | | |")
        r.note("no code files found under the project root; check --root")

    # Which generation of the code is in the root? A V1 root has no
    # answer_question/FactsMode and no attribution.py; a merged V2 root has all
    # three. Saying which one is present prevents a whole class of confusion
    # ("I pasted the new file but the old one still runs").
    root_py = {f["path"]: f for f in r.files if f["path"].count(os.sep) == 0}
    has = lambda name: name in root_py
    talk_path = r.root / "step4_talk.py"
    generation = "unknown"
    if talk_path.is_file():
        try:
            body = talk_path.read_text(encoding="utf-8", errors="replace")
            is_v2 = "def answer_question" in body and "class FactsMode" in body
            generation = ("V2 (patched trial)" if is_v2 else "V1 (original)")
            if is_v2 and not has("attribution.py"):
                r.note("step4_talk.py is the new one but attribution.py is missing beside it; "
                       "copy the whole set, not a single file")
        except OSError:
            pass
    lines += ["", "## Which generation of the code is in the project root", "",
              f"- `step4_talk.py` looks like: **{generation}**",
              f"- `attribution.py` present: **{'yes' if has('attribution.py') else 'no'}** "
              f"(required by the new code, absent from V1)",
              f"- `trial_selftest.py` present: **{'yes' if has('trial_selftest.py') else 'no'}**",
              f"- `start_trial.py` present: **{'yes' if has('start_trial.py') else 'no'}**",
              f"- `v2_trial/` folder present: "
              f"**{'yes' if (r.root / 'v2_trial').is_dir() else 'no'}**",
              "",
              "If this says V1 while you expected V2, the new files are in a subfolder "
              "(or were copied to the wrong place) - that single line explains most "
              "'nothing changed' reports.",
              "", "## Python files in the root, by name", "",
              "This is the answer to 'which version of step4_talk.py am I running?'.",
              ""]
    for f in sorted(r.files, key=lambda x: x["path"]):
        if f["path"].count(os.sep) == 0 and f["path"].endswith(".py"):
            lines.append(f"- `{f['path']}` - {r.human(f['size'])}, {f['mtime']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 3. Metadata CSV
# --------------------------------------------------------------------------
def section_metadata(r: Report) -> tuple[str, list[dict]]:
    csv_path = None
    for cand in (r.root / "bills_data" / "bills_metadata.csv",
                 r.root / "bills_metadata.csv"):
        if cand.is_file():
            csv_path = cand
            break
    if csv_path is None:
        r.note("bills_metadata.csv not found: metadata answers cannot be verified")
        return ("# 3. Metadata\n\n**bills_metadata.csv was not found.**\n", [])

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    lines = ["# 3. Metadata table", "",
             f"- file: `{r.rel(csv_path)}`",
             f"- size: {r.human(csv_path.stat().st_size)}",
             f"- rows: **{len(rows):,}**",
             f"- columns: **{len(columns)}**",
             f"- sha256: `{sha256_of(csv_path)}`", "",
             "## Columns, with blanks", "",
             "| column | blank / missing |", "|---|---:|"]
    for col in columns:
        blank = sum(1 for row in rows if not (row.get(col) or "").strip())
        lines.append(f"| `{col}` | {blank:,} |")

    def counts(col: str, limit: int = 12) -> list[tuple[str, int]]:
        tally: dict[str, int] = {}
        for row in rows:
            v = (row.get(col) or "").strip() or "(blank)"
            tally[v] = tally.get(v, 0) + 1
        return sorted(tally.items(), key=lambda kv: -kv[1])[:limit]

    for col in ("house", "status", "billType", "source_primary"):
        if columns and col in columns:
            lines += ["", f"## `{col}`", "", "| value | rows |", "|---|---:|"]
            for val, n in counts(col):
                lines.append(f"| {val} | {n:,} |")

    # Identity: the three columns that make a bill unique.
    keys, dup_keys = set(), 0
    year_span: set[str] = set()
    for row in rows:
        year = (row.get("billYear") or "").strip()
        no = (row.get("bill_number_norm") or row.get("billNumber") or "").strip()
        house = (row.get("house") or "").strip()
        if year:
            year_span.add(year)
        k = (house, no.upper(), year)
        if k in keys:
            dup_keys += 1
        keys.add(k)

    def year_of(value: str) -> int | None:
        m = re.match(r"^(\d{4})", (value or "").strip())
        return int(m.group(1)) if m else None

    years = sorted({y for y in (year_of(v) for v in year_span) if y})
    outside = sum(1 for row in rows
                  if (year_of(row.get("billYear")) or 0) not in range(2004, 2027))
    no_introducer = sum(1 for row in rows if not (row.get("billIntroducedBy") or "").strip())

    # Introduced dates, so "how fresh is this?" is a measured fact.
    intro: list[str] = []
    for row in rows:
        v = (row.get("billIntroducedDate") or "").strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", v)
        if m:
            intro.append(m.group(1))
    intro.sort()

    # Title year vs billYear: a bill titled "(…Bill, 2021)" may be filed as 2019.
    title_mismatch = 0
    for row in rows:
        ty = YEAR_IN_TITLE.findall(row.get("billName") or "")
        by = (row.get("billYear") or "").strip()
        if ty and by:
            years_in_title = re.findall(r"\b(?:19|20)\d{2}\b", row.get("billName") or "")
            if years_in_title and years_in_title[-1] != by:
                title_mismatch += 1

    lines += ["", "## What the numbers mean", "",
              f"- distinct (house, number, year) keys: **{len(keys):,}**",
              f"- rows repeating an existing key: **{dup_keys:,}** "
              f"(0 means one row per bill)",
              f"- `billYear` range: **{years[0] if years else '?'}-{years[-1] if years else '?'}**",
              f"- rows with `billYear` outside 2004-2026: **{outside:,}**",
              f"- rows with no introducer recorded: **{no_introducer:,}**",
              f"- newest `billIntroducedDate`: **{intro[-1] if intro else 'none'}** "
              f"(this is the data's as-of date)",
              f"- oldest `billIntroducedDate`: **{intro[0] if intro else 'none'}**",
              f"- rows whose title year differs from `billYear`: **{title_mismatch:,}** "
              f"(normal: a February bill carries the previous year's number)"]
    if dup_keys:
        r.note(f"{dup_keys} duplicate (house, number, year) rows in the CSV")
    if not intro:
        r.note("no parseable billIntroducedDate values: freshness cannot be stated")

    lines += ["", "## First 3 rows, as they are stored", "", "```text"]
    for row in rows[:3]:
        shown = {k: (row.get(k) or "")[:60] for k in
                 ("house", "billNumber", "billYear", "billName", "status",
                  "billIntroducedBy", "billIntroducedDate") if k in row}
        lines.append(json.dumps(shown, ensure_ascii=False))
    lines += ["```", ""]
    return "\n".join(lines) + "\n", rows


# --------------------------------------------------------------------------
# 4. Brain
# --------------------------------------------------------------------------
def count_occurrences(path: Path, needle: bytes = b'"chunk_id"') -> int:
    """Count occurrences without loading a multi-hundred-MB file into memory.
    Used for clean_data/chunks.json, which can be over 1 GB."""
    n, carry = 0, b""
    keep = len(needle) - 1
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            buf = carry + block
            n += buf.count(needle)
            carry = buf[-keep:] if keep else b""
    return n


def line_count(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            n += block.count(b"\n")
    return n


def iter_json_list(path: Path, chunk_bytes: int = 4 * 1024 * 1024):
    """Yield the items of a top-level JSON list WITHOUT loading the file.

    kept_chunks.json reached 158 MB in one real project, which is far beyond
    what json.load should be asked to do while a person waits. The array is
    still one JSON document, so raw_decode is used from a moving buffer: each
    item is decoded, the buffer is trimmed, and the rest is read on demand."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8", errors="replace") as f:
        buf = ""
        pos = 0
        # find the opening bracket
        while True:
            more = f.read(chunk_bytes)
            if not more:
                return
            buf += more
            start = buf.find("[")
            if start != -1:
                pos = start + 1
                break
            if len(buf) > 10 * chunk_bytes:
                raise ValueError("no JSON array found in this file")
        while True:
            # skip whitespace and separators
            while True:
                if pos >= len(buf) - 1:
                    more = f.read(chunk_bytes)
                    if not more:
                        break
                    buf = buf[pos:] + more
                    pos = 0
                    continue
                ch = buf[pos]
                if ch in " \t\r\n,":
                    pos += 1
                    continue
                break
            if pos >= len(buf) - 1:
                return
            if buf[pos] == "]":
                return
            try:
                item, end = dec.raw_decode(buf, pos)
            except ValueError:
                more = f.read(chunk_bytes)
                if not more:
                    raise
                buf = buf[pos:] + more
                pos = 0
                continue
            yield item
            pos = end
            if pos > chunk_bytes:
                buf = buf[pos:]
                pos = 0


def brain_bills(chunks: list[dict]) -> dict[tuple[str, str, str], int]:
    out: dict[tuple[str, str, str], int] = {}
    for c in chunks:
        no = str(c.get("bill_no_norm") or c.get("bill_no") or "").strip().upper()
        key = (str(c.get("house") or "").strip(), no, str(c.get("year") or "").strip())
        out[key] = out.get(key, 0) + 1
    return out


def section_brain(r: Report, meta_rows: list[dict]) -> str:
    brain = r.root / "brain"
    lines = ["# 4. Brain (the search index)", ""]
    if not brain.is_dir():
        r.note("brain/ not found: clause questions cannot be answered from an index")
        return "\n".join(lines + ["**brain/ was not found in this project.**", ""]) + "\n"

    # -- files at the top of brain/
    lines += ["## Files", "", "| file | size |", "|---|---:|"]
    for p in sorted(brain.iterdir()):
        if p.is_file():
            lines.append(f"| `{p.name}` | {r.human(p.stat().st_size)} |")
        elif p.is_dir():
            total = sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
            n = sum(1 for q in p.rglob("*") if q.is_file())
            lines.append(f"| `{p.name}/` ({n:,} files) | {r.human(total)} |")

    # -- model + fingerprint
    for name in ("cache_model.txt", "fingerprint.json"):
        p = brain / name
        if p.is_file() and p.stat().st_size < 200_000:
            try:
                body = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                body = "(unreadable)"
            lines += ["", f"## `{name}`", "", "```text", body.strip()[:2000], "```"]

    # -- kept chunks: the heart of it
    kept_path = brain / "kept_chunks.json"
    chunks: list[dict] = []
    if not kept_path.is_file():
        r.note("brain/kept_chunks.json missing: cannot audit index coverage")
        lines += ["", "**brain/kept_chunks.json missing** - coverage cannot be audited.", ""]
        return "\n".join(lines) + "\n"
    # Read it in a STREAMING fashion. A real project produced a 158 MB
    # kept_chunks.json; json.load on that, while the app is also running, is
    # how a laptop starts swapping. Only what the audit needs is kept, and the
    # text itself is reduced to a length and a hash as it goes by.
    KEEP = ("bill_no", "bill_no_norm", "year", "house", "part", "slot",
            "clause_no", "language", "source_primary", "bill_title")
    chunks = []
    text_lens = []
    text_hashes = set()
    duplicates = 0
    empty = 0
    total = 0
    try:
        for item in iter_json_list(kept_path):
            if not isinstance(item, dict):
                continue
            total += 1
            chunks.append({k: item.get(k) for k in KEEP if k in item})
            body = item.get("text") or ""
            if not body.strip():
                empty += 1
            text_lens.append(len(body))
            digest = hash(body.strip())
            if digest in text_hashes:
                duplicates += 1
            else:
                text_hashes.add(digest)
    except Exception as exc:
        r.note(f"kept_chunks.json could not be parsed: {type(exc).__name__}: {exc}")
        return "\n".join(lines + ["", f"**parse failed:** {exc}", ""]) + "\n"
    if not chunks:
        r.note("kept_chunks.json parsed to zero chunks: the index is empty")
        return "\n".join(lines + ["", "**The index contains no chunks.**", ""]) + "\n"

    # One extra bounded pass for the verbatim samples in section 7: the first
    # 300 records that carry text, each cut to 500 characters.
    sample = []
    try:
        for item in iter_json_list(kept_path):
            if isinstance(item, dict) and (item.get("text") or "").strip():
                pick = {k: item.get(k) for k in KEEP}
                pick["text"] = (item.get("text") or "")[:500]
                sample.append(pick)
                if len(sample) >= 300:
                    break
    except Exception:
        pass
    r.sample_chunks = sample

    def tally(field: str, limit: int = 15) -> list[tuple[str, int]]:
        t: dict[str, int] = {}
        for c in chunks:
            v = c.get(field)
            v = "(missing)" if v in (None, "") else str(v)
            t[v] = t.get(v, 0) + 1
        return sorted(t.items(), key=lambda kv: -kv[1])[:limit]

    lengths = sorted(text_lens)
    median = lengths[len(lengths) // 2] if lengths else 0
    dupes = duplicates
    per_bill = brain_bills(chunks)

    lines += ["", "## Size and shape", "",
              f"- chunks in the index: **{len(chunks):,}**",
              f"- distinct bills covered: **{len(per_bill):,}**",
              f"- years covered: **{len({k[2] for k in per_bill})}**",
              f"- chunks with empty text: **{empty:,}**",
              f"- chunks whose text is an exact duplicate of another: **{dupes:,}**",
              f"- text length: shortest {lengths[0] if lengths else 0:,}, "
              f"median {median:,}, longest {lengths[-1] if lengths else 0:,} characters"]

    for field in ("house", "part", "slot", "language", "source_primary"):
        lines += ["", f"## By `{field}`", "", "| value | chunks |", "|---|---:|"]
        for val, n in tally(field):
            lines.append(f"| `{val}` | {n:,} |")

    has_clause = sum(1 for c in chunks if c.get("clause_no"))
    lines += ["", "## Fields the talk layer relies on", "",
              f"- chunks with a clause number: **{has_clause:,}** of {len(chunks):,}",
              f"- chunks with `bill_no_norm`: "
              f"**{sum(1 for c in chunks if c.get('bill_no_norm')):,}** "
              f"(0 means the brain predates that column and falls back to raw numbers)"]
    if not has_clause:
        r.note("no chunk carries a clause number: clause-level answers will be vague")

    # -- coverage: which bills have NO text at all
    missing: list[tuple[str, str, str, str, str]] = []
    zero_text = 0
    if meta_rows:
        brain_keys = set(per_bill)
        for row in meta_rows:
            no = (row.get("bill_number_norm") or row.get("billNumber") or "").strip().upper()
            key = ((row.get("house") or "").strip(), no, (row.get("billYear") or "").strip())
            if key not in brain_keys:
                zero_text += 1
                missing.append((key[0], no, key[2],
                                (row.get("billName") or "")[:90],
                                (row.get("status") or "")))
        covered_pct = 100.0 * (len(meta_rows) - zero_text) / len(meta_rows) if meta_rows else 0.0
        lines += ["", "## Coverage - the important number", "",
                  f"- bills in the metadata table: **{len(meta_rows):,}**",
                  f"- bills with at least one chunk of text: "
                  f"**{len(meta_rows) - zero_text:,}** ({covered_pct:.1f}%)",
                  f"- bills with **no text at all**: **{zero_text:,}**",
                  "",
                  "A question about a bill in the last row of that list can only be "
                  "answered from metadata (dates, status, introducer), never from its "
                  "clauses. That is a data limit, not a code bug - and it is listed in "
                  "`brain-coverage.csv` so each bill can be checked by name."]

    # -- optional: read the Chroma database directly, read-only
    chroma = brain / "chroma_db" / "chroma.sqlite3"
    lines += ["", "## Chroma database (read-only, via sqlite)", ""]
    if chroma.is_file() and chroma.stat().st_size > 2 * 1024 ** 3:
        # A multi-GB database takes minutes to count on a laptop. Its size is
        # already reported above, and the chunk count comes from
        # kept_chunks.json, which is the file the app actually reads. So the
        # count is skipped rather than making a 30-second report a 10-minute one.
        lines.append(f"- `chroma.sqlite3` is {r.human(chroma.stat().st_size)}, "
                     f"too large to count here; skipped on purpose. "
                     f"The authoritative chunk count is in kept_chunks.json above.")
    elif chroma.is_file():
        try:
            uri = f"file:{chroma.as_posix()}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=10)
            try:
                cols = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                lines.append(f"- tables: {', '.join('`%s`' % c for c in cols[:12])}")
                if "collections" in cols:
                    for cid, name in con.execute("SELECT id, name FROM collections"):
                        lines.append(f"- collection `{name}` = `{cid}`")
                if "embeddings" in cols:
                    total = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                    lines.append(f"- rows in `embeddings`: **{total:,}**")
                    if chunks:
                        lines.append(f"- chunks in kept_chunks.json: **{len(chunks):,}**")
                        if total != len(chunks):
                            r.note(f"Chroma holds {total:,} vectors but kept_chunks.json "
                                   f"holds {len(chunks):,}: the brain files disagree, and "
                                   f"step 3 will refuse to load this brain")
            finally:
                con.close()
        except sqlite3.Error as exc:
            lines.append(f"- could not read the database: {type(exc).__name__}: {exc}")
            r.note(f"chroma.sqlite3 unreadable: {type(exc).__name__}: {exc}")
    else:
        lines.append("- `brain/chroma_db/chroma.sqlite3` not found")

    # -- the per-bill CSV
    out_csv = r.out / "brain-coverage.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["house", "bill_number", "bill_year", "chunks_in_brain",
                    "has_text", "status", "title"])
        for row in meta_rows or []:
            no = (row.get("bill_number_norm") or row.get("billNumber") or "").strip().upper()
            key = ((row.get("house") or "").strip(), no, (row.get("billYear") or "").strip())
            n = per_bill.get(key, 0)
            w.writerow([key[0], no, key[2], n, "yes" if n else "no",
                        (row.get("status") or ""), (row.get("billName") or "")[:110]])
        if not meta_rows:
            for key, n in sorted(per_bill.items()):
                w.writerow([key[0], key[1], key[2], n, "yes", "", ""])

    lines += ["", "## Files written next to this one", "",
              "- `brain-coverage.csv` - one row per bill: how many chunks of text the "
              "index holds for it, and `has_text` yes/no.",
              ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 5. clean_data
# --------------------------------------------------------------------------
def section_clean(r: Report) -> str:
    clean = r.root / "clean_data"
    lines = ["# 5. clean_data", ""]
    if not clean.is_dir():
        return "\n".join(lines + ["**clean_data/ was not found.** (The brain can still "
                                  "work; this folder is only the step-2 output.)", ""]) + "\n"
    lines += ["| file | size | rows |", "|---|---:|---:|"]
    for p in sorted(clean.iterdir()):
        if not p.is_file():
            continue
        size = p.stat().st_size
        if p.suffix == ".json":
            n = count_occurrences(p, b'"chunk_id"')
        elif p.suffix == ".csv":
            n = max(0, line_count(p))
        else:
            n = ""
        lines.append(f"| `{p.name}` | {r.human(size)} | {n if n != '' else '-'} |")

    q = clean / "quarantine.csv"
    lines += ["", "## Quarantined files (step 2 could not parse them)", ""]
    if q.is_file() and q.stat().st_size < 20 * 1024 * 1024:
        try:
            with open(q, encoding="utf-8-sig", newline="") as f:
                qrows = list(csv.DictReader(f))
            lines.append(f"- rows: **{len(qrows):,}**")
            if qrows:
                reasons: dict[str, int] = {}
                for row in qrows:
                    reason = (row.get("reason") or row.get("why") or
                              row.get("error") or "(no reason column)")[:120]
                    reasons[reason] = reasons.get(reason, 0) + 1
                lines += ["", "| reason | files |", "|---|---:|"]
                for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:15]:
                    lines.append(f"| {reason} | {n:,} |")
        except Exception as exc:
            lines.append(f"- could not read: {type(exc).__name__}: {exc}")
    else:
        lines.append("- no `quarantine.csv` (nothing failed to parse, or step 2 was not run)")

    cv = clean / "chunks.csv"
    if cv.is_file():
        lines += ["", "Note: `chunks.csv` is a human-readable copy of `chunks.json`. "
                      "No step reads it; it can be deleted to save space without "
                      "affecting the chatbot."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 6. PDFs by stage - counts only, never contents
# --------------------------------------------------------------------------
def section_pdfs(r: Report) -> str:
    pdf_root = r.root / "bills_data" / "pdfs"
    lines = ["# 6. Downloaded documents (counts only)", ""]
    if not pdf_root.is_dir():
        return "\n".join(lines + ["**bills_data/pdfs/ was not found.**", ""]) + "\n"

    per_slot: dict[str, dict] = {}
    exts: dict[str, int] = {}
    unreadable_names = 0
    total_names = 0
    examples: list[str] = []
    for p in pdf_root.rglob("*"):
        if not p.is_file():
            continue
        total_names += 1
        rel = p.relative_to(pdf_root)
        parts = rel.parts
        slot = parts[1] if len(parts) > 2 else "(loose file)"
        agg = per_slot.setdefault(slot, {"files": 0, "bytes": 0})
        agg["files"] += 1
        try:
            agg["bytes"] += p.stat().st_size
        except OSError:
            pass
        exts[p.suffix.lower() or "(none)"] = exts.get(p.suffix.lower() or "(none)", 0) + 1
        if not PDF_NAME.match(p.name):
            unreadable_names += 1
            if len(examples) < 8:
                examples.append(str(rel))

    lines += ["| stage folder | files | size |", "|---|---:|---:|"]
    for slot in sorted(per_slot, key=lambda k: -per_slot[k]["bytes"]):
        a = per_slot[slot]
        lines.append(f"| `{slot}` | {a['files']:,} | {r.human(a['bytes'])} |")
    lines += ["", f"- total files: **{total_names:,}**",
              "", "| extension | files |", "|---|---:|"]
    for ext, n in sorted(exts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{ext}` | {n:,} |")

    lines += ["", "## File names the parser cannot read", "",
              f"- files whose name does not match `<index>_<billno>_<year>_<name>`: "
              f"**{unreadable_names:,}** of {total_names:,}",
              "",
              "Step 2 silently skips these, so their text never reaches the brain. "
              "A non-zero number here explains missing bills better than any guess."]
    if examples:
        lines += ["", "First examples:", ""]
        for ex in examples:
            lines.append(f"- `{ex}`")
    if unreadable_names:
        r.note(f"{unreadable_names:,} downloaded files have names step 2 cannot parse "
               f"and are skipped")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 7. Verbatim samples, so every claim can be traced back
# --------------------------------------------------------------------------
def section_samples(r: Report, chunks: list[dict] | None, samples: int) -> str:
    lines = ["# 7. Verbatim samples", "",
             "Real entries, unedited apart from length, so any answer can be traced.",
             ""]
    if not chunks:
        return "\n".join(lines + ["No chunks available to sample.", ""]) + "\n"
    rnd = random.Random(7)
    picks = rnd.sample(chunks, min(samples, len(chunks)))
    for i, c in enumerate(picks, 1):
        text = re.sub(r"\s+", " ", c.get("text") or "")
        lines += [f"## Sample {i}", "",
                  "```text",
                  f"house      : {c.get('house')}",
                  f"bill       : {c.get('bill_no')} of {c.get('year')}",
                  f"title      : {c.get('bill_title')}",
                  f"part       : {c.get('part')}   clause: {c.get('clause_no')}   "
                  f"slot: {c.get('slot')}",
                  f"source     : {c.get('source_file')}  ({c.get('source_primary')})",
                  f"text       : {text[:420]}{'...' if len(text) > 420 else ''}",
                  "```", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 8. Problems found, in one place
# --------------------------------------------------------------------------
def section_problems(r: Report) -> str:
    lines = ["# 8. Things worth looking at", ""]
    if not r.problems:
        lines += ["Nothing was flagged. That means: no missing CSV, no unreadable brain, "
                  "no disagreeing counts, no unparseable file names.", ""]
    else:
        lines += ["Each line below was produced by an actual check, not a hunch.", ""]
        for i, p in enumerate(r.problems, 1):
            lines.append(f"{i}. {p}")
        lines.append("")
    lines += ["", "## What this report cannot show", "",
              "- Whether an individual PDF's *content* is correct: only counts and names "
              "were read, never the text of a PDF.",
              "- Whether a model-written answer is faithful: that needs a Groq key and a "
              "live question, not a file.",
              "- Anything outside this project folder.",
              ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 9. Ask the project's own code a set of questions (no index, no API key)
# --------------------------------------------------------------------------
def section_answers(r: Report, meta_rows: list[dict]) -> str:
    """Behaviour, not structure: run the real answer function in metadata mode
    and record the path each question takes. This is what shows that a question
    is routed where it should be, and that a refusal is a refusal."""
    lines = ["# 9. Questions asked of this project's own code", "",
             "Each row is the real answer function in the project being inspected, run "
             "with no API key and no search index, so nothing is downloaded and no "
             "model is called. The path is the line the app shows under an answer.",
             ""]
    csv_path = r.root / "bills_data" / "bills_metadata.csv"
    previous_cwd = os.getcwd()
    if csv_path.is_file():
        os.environ.setdefault("BILLS_CSV", str(csv_path))
    os.chdir(r.root)          # the CSV candidates are relative paths
    sys.path.insert(0, str(r.root))
    try:
        import importlib
        talk = importlib.import_module("step4_talk")
    except Exception as exc:
        r.note(f"step4_talk.py could not be imported, so live questions were not run: "
               f"{type(exc).__name__}: {exc}")
        return "\n".join(lines + [f"**Could not import step4_talk.py:** "
                                  f"`{type(exc).__name__}: {exc}`", ""]) + "\n"

    if not hasattr(talk, "answer_question") or not hasattr(talk, "FactsMode"):
        r.note("step4_talk.py has no answer_question/FactsMode: this is an old copy")
        return "\n".join(lines + ["**This copy of step4_talk.py is too old to question "
                                  "(no answer_question/FactsMode).**", ""]) + "\n"

    # Build probes from the data itself, so they are always about real rows.
    probes: list[tuple[str, str]] = []
    if meta_rows:
        def row_year(row):
            try:
                return int(str(row.get("billYear") or 0)[:4])
            except ValueError:
                return 0

        def as_probe(row) -> str:
            no = (row.get("billNumber") or "").strip()
            return (f"Who introduced {(row.get('house') or '').strip()} "
                    f"Bill No. {no} of {(row.get('billYear') or '').strip()}?")

        by_year = sorted(meta_rows, key=row_year)
        if by_year:
            probes.append((as_probe(by_year[0]), "oldest bill in the table"))
            probes.append((as_probe(by_year[-1]), "newest bill in the table"))
        rs = [x for x in meta_rows if "Rajya" in (x.get("house") or "")]
        ls = [x for x in meta_rows if "Lok" in (x.get("house") or "")]
        if rs:
            probes.append((as_probe(rs[-1]), "a Rajya Sabha bill"))
        if ls:
            probes.append((as_probe(ls[len(ls) // 2]), "a Lok Sabha bill, mid-table"))
        no_intro = [x for x in meta_rows if not (x.get("billIntroducedBy") or "").strip()]
        if no_intro:
            probes.append((as_probe(no_intro[0]), "a row with no introducer recorded"))
        titled = [x for x in meta_rows if (x.get("billName") or "").strip()]
        if titled:
            name = re.sub(r"\s+", " ", titled[len(titled) // 3].get("billName") or "").strip()
            probes.append((f"Tell me about {name}", "a bill named by its title"))
            probes.append((f"What does {name} say?", "a text question with no index"))

    probes += [
        ("Who proposed this bill?", "a follow-up with no bill yet named"),
        ("What is the capital of France?", "off-topic"),
        ("Tell me about Bill No. 999 of 2019.", "a bill number that does not exist"),
        ("Tell me about Bill No. 370 of 1999.", "a year outside 2004-2026"),
        ("How many bills were introduced in 2019?", "a count"),
        ("Which bills were passed in December 2019?", "a date list"),
        ("Hello", "greeting"),
    ]

    try:
        return _run_probes(r, talk, meta_rows, probes, lines)
    finally:
        os.chdir(previous_cwd)


def _run_probes(r: Report, talk, meta_rows, probes, lines) -> str:
    """Run each probe and record the path and text. Split out so the caller can
    restore the working directory in a finally block."""
    brain = talk.FactsMode()
    rows_out = []
    for question, why in probes:
        try:
            reply, hits, used = talk.answer_question(brain, question, no_llm=True)
        except Exception as exc:
            r.note(f"answering {question[:40]!r} raised {type(exc).__name__}: {exc}")
            rows_out.append((question, why, f"RAISED {type(exc).__name__}", str(exc)[:160]))
            continue
        flat = re.sub(r"\s+", " ", str(reply)).strip()
        rows_out.append((question, why, str(used), flat[:260]))

    lines += ["| question | why this probe | answer path |", "|---|---|---|"]
    for question, why, used, _ in rows_out:
        lines.append(f"| `{question[:110]}` | {why} | `{used}` |")
    lines += ["", "## Full text of those answers", ""]
    for question, why, used, flat in rows_out:
        lines += [f"**Q:** {question}  ", f"*{why} · path `{used}`*", "",
                  "```text", flat or "(empty)", "```", ""]

    paths = {used for _, _, used, _ in rows_out}
    raised = [u for u in paths if u.startswith("RAISED")]
    if raised:
        r.note(f"{len(raised)} probe(s) raised an exception instead of answering")
    bad = [q for q, _, used, flat in rows_out
           if used in ("refusal", "error") and "capital of France" not in q]
    if bad:
        r.note(f"{len(bad)} ordinary question(s) came back as {bad[0][:40]!r}...")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def find_root(start: Path) -> Path | None:
    for base in [start, *start.parents][:5]:
        if (base / "bills_data" / "bills_metadata.csv").is_file():
            return base
        if (base / "brain").is_dir() and (base / "step4_talk.py").is_file():
            return base
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect a small, complete report of this project.")
    ap.add_argument("--root", default="", help="project folder (default: search upward from cwd)")
    ap.add_argument("--out", default="", help="output folder (default: <root>/_report_<stamp>)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--max-mb", type=float, default=8.0)
    args = ap.parse_args()

    start = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()
    root = find_root(start) if not args.root else start
    if root is None or not root.is_dir():
        print("Could not find the project folder.\n"
              "Looked for bills_data\\bills_metadata.csv from:\n  " + str(start) + "\n\n"
              "Run it from the project folder, or pass --root.", file=sys.stderr)
        return 2

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    out = Path(args.out).expanduser().resolve() if args.out else root / f"_report_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    r = Report(root, out)
    print(f"Project : {root}")
    print(f"Output  : {out}")
    print(f"Python  : {sys.executable}")

    print("[1/9] environment ...")
    r.write("01-ENVIRONMENT.md", section_environment(r))
    print("[2/9] files ...")
    collect_files(r)
    r.write("02-FILES.md", section_files(r))
    print("[3/9] metadata ...")
    meta_text, meta_rows = section_metadata(r)
    r.write("03-METADATA.md", meta_text)
    print("[4/9] brain ...")
    r.write("04-BRAIN.md", section_brain(r, meta_rows))

    # Samples are read from the SAME streaming pass the brain section uses;
    # re-parsing a 158 MB file a second time would double the time for nothing.
    chunks = getattr(r, "sample_chunks", None)

    print("[5/9] clean_data ...")
    r.write("05-CLEAN-DATA.md", section_clean(r))
    print("[6/9] documents ...")
    r.write("06-DOCUMENTS.md", section_pdfs(r))
    print("[7/9] samples ...")
    r.write("07-SAMPLES.md", section_samples(r, chunks, max(0, args.samples)))
    print("[8/9] questions ...")
    r.write("09-ANSWERS.md", section_answers(r, meta_rows))
    print("[9/9] problems ...")
    r.write("08-PROBLEMS.md", section_problems(r))

    # ---- size guard, then zip
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    limit = args.max_mb * 1024 * 1024
    if total > limit:
        r.note(f"report folder is {r.human(total)}, above the {args.max_mb} MB cap; "
               f"re-run with --samples 5 to shrink it")
        r.write("08-PROBLEMS.md", section_problems(r))

    zip_path = root / f"report-{stamp}.zip"
    if not args.no_zip:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                if p.is_file():
                    z.write(p, p.name)
        size = zip_path.stat().st_size
        print("\nWrote:")
        for p in sorted(out.iterdir()):
            if p.is_file():
                print(f"  {p.name:<22} {p.stat().st_size/1024:8.1f} KB")
        print(f"\nZIP : {zip_path}")
        print(f"SIZE: {size/1024:.1f} KB ({size/1048576:.2f} MB)")
        print("\nSend that ZIP. Nothing outside the output folder was modified.")
    else:
        print(f"\nFolder written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
