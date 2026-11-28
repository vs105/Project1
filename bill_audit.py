#!/usr/bin/env python3
"""
bill_audit.py - examine EVERY bill, not a sample.

Why this exists
---------------
The other suites check the engine deeply on a few hundred questions. This one
answers a different, simpler question: "for EACH of my 5,169 bills, will the
chatbot answer the basic questions correctly - and if not, WHICH bill and WHY?"

For every bill in bills_metadata.csv it asks four questions a teacher would ask:

  1. IDENTITY  "Tell me about <House> Bill No. N of YYYY"   -> right bill named?
  2. WHO       "Who introduced <House> Bill No. N of YYYY?" -> the CSV's name,
                 or the ministry if no name is recorded
  3. STATUS    "Is <House> Bill No. N of YYYY <status>?"    -> the CSV's status
  4. WHEN      "When was <House> Bill No. N of YYYY introduced?" -> the CSV's date

It also records whether the brain holds any TEXT for the bill (readable vs
metadata-only), so the report can say which bills cannot answer clause
questions and why (no document downloaded, or document unreadable/scanned).

Output (written next to this script, plain text + a spreadsheet):

  BILL-AUDIT.txt      summary + EVERY failure with its reason (small)
  BILL-AUDIT-ALL.csv  one row per bill, every question's verdict (complete)

Run:

    .\\.venv\\Scripts\\python.exe bill_audit.py              # all 5,169 bills
    .\\.venv\\Scripts\\python.exe bill_audit.py --sample 200 # quick look first

Nothing is downloaded, no model is called, no project file is modified.
Metadata questions only (facts mode), so it is fast: about 1-3 minutes for all
bills on a normal laptop.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
STATUS_ROOTS = {"assented": "assent", "pending": "pending", "withdrawn": "withdraw",
                "lapsed": "laps", "passed": "pass", "rejected": "reject",
                "removed": "remov", "negatived": "negativ"}
DOC_COLUMNS = [("billIntroducedFile", "intro"), ("billPassedInLSFile", "passedLS"),
               ("billPassedInRSFile", "passedRS"), ("billPassedInBothHousesFile", "passedBoth"),
               ("billGazettedFile", "gazette"), ("billSynopsisFile", "synopsis"),
               ("errataFile", "errata"), ("reportFile", "report")]


def find_csv() -> Path | None:
    for cand in (HERE / "bills_data" / "bills_metadata.csv",
                 HERE.parent / "bills_data" / "bills_metadata.csv"):
        if cand.is_file():
            return cand
    env = os.environ.get("BILLS_CSV")
    return Path(env) if env and Path(env).is_file() else None


def parse_day(value):
    """BillsDB stores dates as datetime.date; accept a raw dd/mm/yyyy string too.
    Returns (day, month, year) or None."""
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value.day, value.month, value.year
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(value or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def brain_text_counts() -> dict:
    """bill key -> chunks of indexed text, streamed from brain/kept_chunks.json.
    Empty dict if the brain is absent or unreadable."""
    kept = HERE / "brain" / "kept_chunks.json"
    if not kept.is_file():
        kept = HERE.parent / "brain" / "kept_chunks.json"
    if not kept.is_file():
        return {}
    try:
        from make_report import iter_json_list
    except Exception:
        return {}
    counts: dict = {}
    try:
        for item in iter_json_list(kept):
            if not isinstance(item, dict):
                continue
            if not (item.get("text") or "").strip():
                continue
            key = (str(item.get("house") or ""), str(item.get("bill_no") or ""),
                   str(item.get("year") or ""))
            counts[key] = counts.get(key, 0) + 1
    except Exception:
        return {}
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask the standard battery for every bill.")
    ap.add_argument("--sample", type=int, default=0, help="only the first N bills")
    ap.add_argument("--out", default=str(HERE), help="where to write the two result files")
    args = ap.parse_args()

    csv_path = find_csv()
    if csv_path is None:
        print("bills_metadata.csv not found. Run from the project folder or set BILLS_CSV.")
        return 2

    import step4_talk as talk                       # noqa: E402
    from bills_db import BillsDB                    # noqa: E402

    db = BillsDB(str(csv_path))
    bills = db.bills[:args.sample] if args.sample else db.bills
    total = len(bills)

    print(f"Bill audit\n  csv    : {csv_path}\n  bills  : {total:,}"
          f"{' (sample)' if args.sample else ''}\n")

    # ---- readability: which bills the brain holds text for -----------------
    print("  reading brain/kept_chunks.json (streaming) ...", end="", flush=True)
    t0 = time.time()
    text_counts = brain_text_counts()
    print(f" {time.time() - t0:.1f}s, {len(text_counts):,} bills have text"
          if text_counts else " none found (metadata-only audit)")

    def has_text(bill) -> str:
        if not text_counts:
            return "unknown"
        for key in ((bill["house"], bill["no"], bill["year"]),
                    (bill["house"], bill["norm"], bill["year"])):
            if key in text_counts:
                return "yes"
        return "no"

    # ---- the battery -------------------------------------------------------
    def ask(brain, q):
        reply, _hits, used = talk.answer_question(brain, q, no_llm=True)
        return re.sub(r"\s+", " ", str(reply)).strip(), used

    rows = []
    fails = []
    t0 = time.time()
    for i, bill in enumerate(bills, 1):
        house, no, year = bill["house"], bill["no"], bill["year"]
        ident = f"{house} Bill No. {no} of {year}"
        brain = talk.FactsMode()                    # fresh: no context bleed
        verdict = {"identity": "", "who": "", "status": "", "when": ""}
        notes = []

        # 1. IDENTITY - the answer must be about THIS bill
        text, used = ask(brain, f"Tell me about {ident}")
        if f"{no} of {year}" in text:
            verdict["identity"] = "PASS"
        else:
            verdict["identity"] = "FAIL"
            notes.append(f"identity: asked {ident}, got [{used}] {text[:110]}")

        # 2. WHO - recorded name, else ministry, else 'no individual'
        text, used = ask(brain, f"Who introduced {ident}?")
        introducer = (bill["by"] or "").strip()
        ministry = (bill.get("ministry") or "").strip()
        if introducer and introducer.casefold() in text.casefold():
            verdict["who"] = "PASS"
        elif not introducer and (ministry and ministry.casefold() in text.casefold()
                                 or "no individual" in text.casefold()
                                 or "signatory" in text.casefold()):
            verdict["who"] = "PASS"
        else:
            verdict["who"] = "FAIL"
            notes.append(f"who: wanted '{introducer or ministry or 'no individual'}', "
                         f"got [{used}] {text[:110]}")

        # 3. STATUS - the CSV's status, by word root (assent date answers count)
        status = (bill["status"] or "").strip()
        if not status:
            verdict["status"] = "SKIP"
        else:
            text, used = ask(brain, f"Is {ident} {status.casefold()}?")
            if STATUS_ROOTS.get(status.casefold(), status.casefold()) in text.casefold():
                verdict["status"] = "PASS"
            else:
                verdict["status"] = "FAIL"
                notes.append(f"status: wanted '{status}', got [{used}] {text[:110]}")

        # 4. WHEN - day + month name + year of the introduction date
        dmy = parse_day(bill.get("intro") or "")
        if not dmy:
            verdict["when"] = "SKIP"
        else:
            d, m, y = dmy
            text, used = ask(brain, f"When was {ident} introduced?")
            low = text.casefold()
            if str(y) in text and MONTHS[m].casefold() in low and re.search(rf"\b{d}\b", text):
                verdict["when"] = "PASS"
            else:
                verdict["when"] = "FAIL"
                notes.append(f"when: wanted {d} {MONTHS[m]} {y}, got [{used}] {text[:110]}")

        readable = has_text(bill)
        rows.append({"house": house, "bill_number": no, "year": year,
                     "type": bill.get("type", ""), "category": bill.get("category", ""),
                     "status": status, "has_text": readable,
                     "identity": verdict["identity"], "who": verdict["who"],
                     "status_check": verdict["status"], "when": verdict["when"],
                     "title": bill["title"]})
        if notes:
            fails.append((ident, bill["title"], notes))

        if i % 500 == 0 or i == total:
            rate = i / max(time.time() - t0, 0.01)
            print(f"  {i:,}/{total:,} bills  ({rate:,.0f}/s, "
                  f"{len(fails)} failing so far)", flush=True)

    # ---- write the complete spreadsheet ------------------------------------
    out_dir = Path(args.out)
    all_csv = out_dir / "BILL-AUDIT-ALL.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- write the summary + every failure ---------------------------------
    def tally(col):
        p = sum(1 for r in rows if r[col] == "PASS")
        f = sum(1 for r in rows if r[col] == "FAIL")
        s = sum(1 for r in rows if r[col] == "SKIP")
        return p, f, s

    no_text = [r for r in rows if r["has_text"] == "no"]
    no_text_govt = sum(1 for r in no_text if r["type"] == "Government")
    no_text_ls = sum(1 for r in no_text if r["house"] == "Lok Sabha")

    lines = []
    add = lines.append
    add("BILL AUDIT - every bill, four standard questions")
    add(f"generated: {datetime.now():%Y-%m-%d %H:%M}")
    add(f"csv: {csv_path}")
    add("")
    add("== INVENTORY ==")
    from collections import Counter
    add(f"bills: {total:,}")
    for k, v in Counter(r["house"] for r in rows).most_common():
        add(f"  {v:>6,}  {k}")
    for k, v in Counter(r["type"] for r in rows).most_common():
        add(f"  {v:>6,}  {k} bills")
    add("")
    add("== READABILITY (can it answer CLAUSE questions, not just dates) ==")
    if text_counts:
        add(f"  bills with text in the brain : {total - len(no_text):,}")
        add(f"  bills with NO text           : {len(no_text):,} "
            f"(Government {no_text_govt}, Private Member {len(no_text) - no_text_govt}; "
            f"Lok Sabha {no_text_ls}, Rajya Sabha {len(no_text) - no_text_ls})")
        add("  -> these answer dates/status/introducer only, never clauses.")
        add("     Reasons: no document downloaded, or a scanned PDF (NEEDS_OCR).")
        add("     The row-level list is BILL-AUDIT-ALL.csv, column has_text = no.")
    else:
        add("  brain not found - readability not measured (metadata answers unaffected).")
    add("")
    add("== QUESTION BATTERY ==")
    for col, label in (("identity", "1. identity  (Tell me about...)"),
                       ("who", "2. who       (Who introduced...)"),
                       ("status_check", "3. status    (Is it <status>?)"),
                       ("when", "4. when      (When introduced?)")):
        p, f, s = tally(col)
        skip = f", {s} skipped (field blank in CSV)" if s else ""
        add(f"  {label:<38} PASS {p:>6,}   FAIL {f:>5,}{skip}")
    add("")
    if fails:
        add(f"== FAILURES - {len(fails)} bill(s), each with the reason ==")
        for ident, title, notes in fails:
            add("")
            add(f"* {ident} - {title}")
            for n in notes:
                add(f"    {n}")
    else:
        add("== FAILURES ==  none. Every bill answered all four questions correctly.")
    add("")
    add(f"Complete row-level record: {all_csv.name}")
    txt = out_dir / "BILL-AUDIT.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n  wrote {txt.name}  ({txt.stat().st_size / 1024:.0f} KB)")
    print(f"  wrote {all_csv.name} ({all_csv.stat().st_size / 1024:.0f} KB, {total:,} rows)")
    print(f"\n  RESULT: {total - len(fails):,} of {total:,} bills passed all four questions"
          f" - {len(fails)} need a look (reasons in BILL-AUDIT.txt)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
