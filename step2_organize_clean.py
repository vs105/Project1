"""
STEP 2 - ORGANISE AND CLEAN

Reads the downloaded bill PDFs and turns them into clean text chunks.

What is new in this version, and why:

1. Every chunk is tagged with a PART. Indian bill PDFs hold two completely
   different kinds of writing. Before the enacting formula is the Statement of
   Objects and Reasons, which is the minister explaining why the bill exists.
   After it are the clauses, which are the actual law. The old chunker mixed
   them, so a question about a punishment could be answered from the
   minister's background note instead of the section. Now the two are split
   apart and tagged, so the talk layer can ask for the right one.

2. Chunks follow CLAUSES, not fixed word windows. A clause is the smallest
   thing that means something on its own. Splitting a clause across two chunks
   broke references like "the aforesaid Bill", because the thing it referred to
   was in the other chunk. Now a clause stays whole, and only a very long
   clause is split, at its own sub-clause letters.

3. DUPLICATES ACROSS FILES ARE REMOVED. Each bill is downloaded as up to eight
   PDFs, and the same clause text appears in several of them. In the old build
   61 percent of the brain was the same clause stored four to eight times,
   which crowded the search results. Now identical text is stored once, and the
   other files it came from are recorded in "also_in", so nothing is lost.

4. Unreadable files are never skipped silently. They go to quarantine.csv with
   the reason, so the talk layer can tell the user "this bill's file is a scan"
   instead of just failing to answer.

Run it like this:
    python step2_organize_clean.py --in ./bills_data --out ./clean_data
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata

# ---------------------------------------------------------------- settings ---

MIN_CHUNK_WORDS = 40      # shorter than this is noise (a heading, a page number)
MIN_CLAUSE_WORDS = 12     # a real clause can be one sentence, so keep short ones
MAX_CHUNK_WORDS = 900     # longer than this and one clause is too big to embed
SUB_SPLIT_WORDS = 700     # a clause longer than this is split at its (a)(b)(c)
NARRATIVE_MAX = 400       # the minister's explanation is chunked smaller

# Lines that are page furniture, not bill text. If these are left in, a page
# footer like "2019." gets read as clause number 2019 and swallows the real
# clause that follows it. That is how clause 3 of the Triple Talaq bill went
# missing from an earlier build.
NOISE_LINE = re.compile(
    r"^\s*("
    r"\*{3,}|_{3,}|-{3,}|={3,}|~{3,}"
    r"|AS\s+INTRODUCED|AS\s+PASSED\s+BY|AS\s+REPORTED\s+BY"
    r"|LOK\s+SABHA|RAJYA\s+SABHA|SECRETARIAT|PARLIAMENT\s+OF\s+INDIA"
    r"|PRINTED\s+BY|PUBLISHED\s+BY|GOVERNMENT\s+OF\s+INDIA\s+PRESS"
    r"|www\.|http|MGIP|BNS\d|GI\(S3\)|LS\(S3\)|RS\(S3\)"
    r"|\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{4}"
    r"|(19|20)\d{2}\s*\.?\s*$"
    r")\s*$",
    re.IGNORECASE)

# A line that is nothing but a number is a page number.
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")

# The eight slots the downloader saves. Only these are looked for.
SLOT_DIRS = {
    "billIntroducedFile": "intro",
    "billPassedInLSFile": "passedLS",
    "billPassedInRSFile": "passedRS",
    "billPassedInBothHousesFile": "passedBoth",
    "errataFile": "errata",
    "reportFile": "committeeReport",
    "billGazettedFile": "gazette",
    "billSynopsisFile": "synopsis",
}

# The words that mark where the minister's explanation ends and the law begins.
ENACTING = re.compile(
    r"BE\s+it\s+enacted\s+by\s+Parliament", re.IGNORECASE)

# In these official PDFs the explanatory material is APPENDED AFTER the clause
# text, not put before it. The Statement of Objects of the Bharatiya Nyaya
# Sanhita sits at 86 percent of the file while the enacting formula sits at 5
# percent. So the operative region runs from the formula up to the first of
# these headings that follows it, and everything after that is narrative.
NARRATIVE_MARKERS = [
    r"STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS",
    r"NOTES\s+ON\s+CLAUSES",
    r"FINANCIAL\s+MEMORANDUM",
    r"MEMORANDUM\s+REGARDING\s+DELEGATED\s+LEGISLATION",
    r"ARRANGEMENT\s+OF\s+CLAUSES",
]

# A clause number at the start of a line, like "12." or "158B."
CLAUSE_HEAD = re.compile(r"^\s*(\d{1,4}[A-Z]?)\.\s+(?=\S)", re.MULTILINE)

# A chapter or schedule heading, like "CHAPTER XIV-B" or "SCHEDULE I"
TERM_HEAD = re.compile(
    r"^\s*(CHAPTER\s+[IVXLC]+[A-Z\-]*|SCHEDULE\s+[IVXLC]+)"
    r"\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE)

# ------------------------------------------------------------------- helpers --

def norm_text(s):
    """Clean up odd characters that PDF extraction leaves behind."""
    s = unicodedata.normalize("NFKD", s or "")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # Soft hyphens and zero width spaces break words in half.
    s = s.replace("\u00ad", "").replace("\u200b", "")
    return s


def looks_english(text):
    """Rough language check. Most bills are English, a few are Hindi."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum(1 for c in letters if "\u0900" <= c <= "\u097f")
    return devanagari / len(letters) < 0.3


def read_pdf(path):
    """Pull text out of a PDF. Returns None if it is a scan or unreadable."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, "NO_PYMUPDF: pip install pymupdf"
    try:
        doc = fitz.open(path)
    except Exception as e:
        return None, "UNREADABLE: " + type(e).__name__
    try:
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        text = "\n".join(pages)
    finally:
        doc.close()
    if len(text.split()) < 20:
        # A scan gives almost no text. This is the OCR case.
        return None, "NEEDS_OCR: no text layer, file is a scanned image"
    return norm_text(text), None


def find_libreoffice():
    """
    Locate the LibreOffice command line program.

    It is called "libreoffice" on Linux and Mac, but "soffice.exe" on Windows,
    where the installer also leaves it out of PATH. Calling only the Linux name
    quarantined 408 real .doc files on a Windows machine that had LibreOffice
    installed perfectly well, which is why this looks in three places.
    """
    import shutil
    for name in ("libreoffice", "soffice"):
        found = shutil.which(name)
        if found:
            return found
    if os.name == "nt":
        roots = (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        for root in roots:
            for name in ("soffice.exe", "libreoffice.exe"):
                p = os.path.join(root, "LibreOffice", "program", name)
                if os.path.exists(p):
                    return p
    return None


_LO_PROFILE = [None]


def read_doc(path):
    """Old .doc and .docx files need LibreOffice, not PyMuPDF."""
    import subprocess
    exe = find_libreoffice()
    if not exe:
        return None, ("DOC_NO_LIBREOFFICE: libreoffice not found - install it "
                      "from libreoffice.org, then open a NEW terminal")
    outdir = os.path.dirname(path)
    # A private profile stops a conversion failing just because the user has
    # LibreOffice open, and stops two conversions fighting over one profile.
    if _LO_PROFILE[0] is None:
        import pathlib, tempfile
        _LO_PROFILE[0] = pathlib.Path(tempfile.mkdtemp(prefix="bills_lo_")).as_uri()
    cmd = [exe, "--headless", "--norestore",
           "-env:UserInstallation=" + _LO_PROFILE[0],
           "--convert-to", "txt:Text", "--outdir", outdir, path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except subprocess.CalledProcessError as e:
        # LibreOffice exits non-zero but prints the real reason on stderr.
        err = (e.stderr or b"").decode("utf-8", "replace").strip()[:200]
        return None, "DOC_NO_LIBREOFFICE: conversion failed - " + (err or "no message")
    except Exception as e:
        return None, "DOC_FAILED: " + type(e).__name__
    produced = os.path.join(outdir,
                            os.path.splitext(os.path.basename(path))[0] + ".txt")
    if os.path.exists(produced):
        with open(produced, encoding="utf-8", errors="replace") as f:
            return norm_text(f.read()), None
    return None, "DOC_NO_LIBREOFFICE: conversion produced no file"


def read_any(path):
    """Read a downloaded file whatever its extension is."""
    low = path.lower()
    if low.endswith(".pdf"):
        return read_pdf(path)
    if low.endswith((".doc", ".docx", ".rtf")):
        return read_doc(path)
    return None, "OTHER_FORMAT: " + os.path.splitext(path)[1]


# ------------------------------------------------------- splitting into parts --

def split_at_enacting(text):
    """
    Work out which part of a bill is the actual law and which part is the
    minister explaining it.

    The law starts at the enacting formula. It stops at the first explanatory
    heading that follows the formula, because in these official PDFs the
    Statement of Objects and the Notes on Clauses are printed AFTER the clause
    text, not before it. Splitting only at the formula would put the minister's
    notes on the law side, which is how a question about a punishment once got
    answered from background commentary.

    Returns (narrative_text, operative_text).
    """
    m = ENACTING.search(text)
    if not m:
        # No formula at all. Some short files hold nothing but clauses.
        return "", text

    start = m.start()
    end = len(text)
    for pat in NARRATIVE_MARKERS:
        mm = re.search(pat, text[start:], re.IGNORECASE)
        if mm and start + mm.start() < end:
            end = start + mm.start()

    narrative = text[:start] + text[end:]
    operative = text[start:end]
    return narrative, operative


def strip_noise(text):
    """
    Take out page furniture before looking for clause numbers.

    Without this, a footer line like "2019." is read as clause number 2019, and
    everything up to the next real clause gets swallowed into that fake chunk.
    That is how clause 3 of the Triple Talaq bill disappeared from a build.
    """
    lines = text.split("\n")
    keep = [ln for ln in lines
            if not NOISE_LINE.match(ln) and not PAGE_NUMBER.match(ln)]
    return "\n".join(keep)


def parse_operative(text):
    """
    Turn the clause half of a bill into a list of dicts, one per clause.
    Each dict has: clause_no, term, label, body.
    """
    text = strip_noise(text)
    marks = list(CLAUSE_HEAD.finditer(text))
    if not marks:
        return [{"clause_no": "", "term": "", "label": "", "body": text.strip()}]

    # Find chapter and schedule headings and where they sit, so each clause can
    # be told which chapter it belongs to.
    terms = []
    for m in TERM_HEAD.finditer(text):
        terms.append((m.start(), m.group(1).strip().upper()))

    # A number between 1900 and 2100 followed by a full stop is a year in a page
    # footer, never a clause number. No bill has two thousand clauses. The
    # isdigit check matters because clause numbers can be like "148A".
    marks = [m for m in marks
             if not (m.group(1).isdigit()
                     and 1900 <= int(m.group(1)) <= 2100
                     and len(m.group(1)) == 4)]

    out = []
    for i, m in enumerate(marks):
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[start:end].strip()

        # Which chapter does this clause sit in
        term = ""
        for pos, name in terms:
            if pos < start:
                term = name
            else:
                break

        # The margin heading, if there is one. In extracted text it usually
        # lands on its own line either just before or just after the number.
        label = find_label(text, start, end)

        out.append({"clause_no": m.group(1), "term": term,
                    "label": label, "body": body})
    return out


def find_label(text, start, end):
    """
    Look for the short heading line that goes with a clause, like
    "Short title, extent and commencement." PDF extraction puts these margin
    notes in odd places, so check the lines just before and just after.
    """
    def is_label(line):
        s = line.strip()
        if not (3 <= len(s) <= 70):
            return False
        if s.endswith((",", ";", ":")) and len(s) < 25:
            return False
        words = s.split()
        if len(words) > 9:
            return False
        # A heading has no verbs and no bracket letters
        if re.search(r"\b(shall|may|is|are|be|was|were|has|have|provided)\b", s, re.I):
            return False
        if re.match(r"^\(?[a-z]\)", s):
            return False
        return s[0].isupper() or s[0].isdigit()

    before = text[max(0, start - 260):start].split("\n")
    for line in reversed(before[-6:]):
        if NOISE_LINE.match(line):
            continue
        if is_label(line):
            return line.strip()
    after = text[end:end + 260].split("\n")
    for line in after[:6]:
        if NOISE_LINE.match(line):
            continue
        if is_label(line):
            return line.strip()
    return ""


def split_long_clause(clause, limit):
    """
    A clause longer than the limit is split at its own sub-clause letters
    (a) (b) (c), never in the middle of a sentence. If it has no sub-clauses
    it is left whole rather than butchered.
    """
    body = clause["body"]
    if len(body.split()) <= limit:
        return [clause]
    marks = list(re.finditer(r"^\s*\(([a-z]{1,2})\)\s+", body, re.MULTILINE))
    if len(marks) < 2:
        return [clause]
    parts = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        piece = body[m.start():end].strip()
        if len(piece.split()) < MIN_CHUNK_WORDS:
            continue
        parts.append({"clause_no": clause["clause_no"], "term": clause["term"],
                      "label": clause["label"], "body": piece})
    return parts or [clause]


def build_chunks(text, meta, slot, source_file):
    """Turn one bill file into a list of chunk records."""
    before, after = split_at_enacting(text)
    records = []

    # The narrative half, kept as its own chunks so "why was this bill brought"
    # questions can be answered, but never used for provision questions.
    if before.strip():
        for piece in paragraphs(strip_noise(before), NARRATIVE_MAX):
            if len(piece.split()) >= MIN_CHUNK_WORDS:
                records.append(make_record(piece, meta, slot, source_file,
                                           "narrative", "", "", ""))

    for clause in parse_operative(after):
        for piece in split_long_clause(clause, SUB_SPLIT_WORDS):
            body = piece["body"]
            if len(body.split()) < MIN_CLAUSE_WORDS:
                continue
            records.append(make_record(body, meta, slot, source_file,
                                       "operative", piece["clause_no"],
                                       piece["term"], piece["label"]))
    return records


def paragraphs(text, cap=MAX_CHUNK_WORDS):
    """Split a block of prose into paragraph sized pieces."""
    blocks = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len((buf + " " + b).split()) > cap and buf:
            out.append(buf)
            buf = b
        else:
            buf = (buf + "\n\n" + b).strip()
    if buf:
        out.append(buf)
    return out


def make_record(body, meta, slot, source_file, part, clause_no, term, label):
    """One chunk record. The field list is fixed so step 3 can rely on it."""
    body = body.strip()
    return {
        "chunk_id": "",                      # filled in later, must be unique
        "bill_no": str(meta.get("billNumber", "")).strip(),
        "year": str(meta.get("billYear", "")).strip(),
        "bill_title": (meta.get("billName") or "").strip(),
        "bill_type": (meta.get("billType") or "").strip(),
        "ministry": (meta.get("ministryName") or "").strip(),
        "status": (meta.get("status") or "").strip(),
        "slot": slot,
        "source_file": source_file,
        "language": "english" if looks_english(body) else "hindi",
        "part": part,                        # operative, narrative
        "clause_no": clause_no,
        "term": term,
        "label": label,
        "words": len(body.split()),
        "text": body,
    }


# ------------------------------------------------------------------- dedupe --

def dedupe(records):
    """
    Keep one copy of each piece of text per bill.

    The same clause is printed in the introduced file, the LS passed file, the
    both houses file and the gazette. Those are the same words, so keeping all
    four just crowds the search results. One copy is kept and the other files
    it came from are listed in also_in, so nothing is actually lost.

    If two chunks have the same bill and clause number but DIFFERENT text, both
    are kept, because that is a real difference between the introduced and the
    passed version of a clause.
    """
    kept, index = [], {}
    for r in records:
        head = " ".join(r["text"].lower().split())[:200]
        key = (r["bill_no"], r["year"], r["part"], r["clause_no"], head)
        if key in index:
            first = index[key]
            first["also_in"].append(r["slot"])
            if len(r["text"]) > len(first["text"]):
                # keep the fuller copy but remember where the shorter was
                r["also_in"] = first["also_in"]
                kept[kept.index(first)] = r
                index[key] = r
            continue
        r["also_in"] = [r["slot"]]
        index[key] = r
        kept.append(r)
    return kept


# ---------------------------------------------------------------------- main --

def find_metadata(bills_dir):
    """Load the bill metadata the downloader saved next to the PDFs."""
    for name in ("bills_metadata.csv", "metadata.csv"):
        p = os.path.join(bills_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return {str(r.get("billNumber")) + "|" + str(r.get("billYear")): r
                        for r in csv.DictReader(f)}
    return {}


def quarantine_row(fn, reason, detail, m):
    """
    One quarantined file. The bill number and year are carried across so the
    user can see WHICH BILLS lost text, not only which files failed. Both are
    also inside the file name, but a CSV column needs no parsing.
    """
    return {"file": fn, "reason": reason, "detail": detail,
            "bill_no": str(m.get("billNumber", "")).strip(),
            "year": str(m.get("billYear", "")).strip()}


def main():
    ap = argparse.ArgumentParser(description="Step 2: PDFs to clean chunks")
    ap.add_argument("--in", dest="indir", default="./bills_data")
    ap.add_argument("--out", dest="outdir", default="./clean_data")
    ap.add_argument("--max-files", type=int, default=0,
                    help="0 means every file")
    args = ap.parse_args()

    if not os.path.isdir(args.indir):
        sys.exit("\nERROR: no input folder " + args.indir +
                 "\nRun download_bills.py first.\n")

    meta = find_metadata(args.indir)
    if not meta:
        sys.exit("\nERROR: no bills_metadata.csv in " + args.indir +
                 "\nRun download_bills.py first.\n")

    # Collect every downloaded file with the bill it belongs to.
    jobs = []
    pdf_root = os.path.join(args.indir, "pdfs")
    for root, _dirs, files in os.walk(pdf_root):
        for fn in sorted(files):
            if not fn.lower().endswith((".pdf", ".doc", ".docx")):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, pdf_root)
            slot_dir = rel.split(os.sep)[1] if os.sep in rel else ""
            slot = next((s for s, d in SLOT_DIRS.items() if d == slot_dir), "")
            if not slot:
                continue
            # File names are  <index>_<billno>_<year>_<name>.pdf
            # so the bill number is the SECOND field and the year the third.
            m = re.match(r"\d+_(\d+)_(\d{4})_", fn)
            if not m:
                continue
            key = m.group(1) + "|" + m.group(2)
            jobs.append((path, slot, fn, meta.get(key, {})))
    if args.max_files:
        jobs = jobs[:args.max_files]

    # A silent metadata miss is the worst possible failure: every chunk would
    # come out with an empty bill number, so the bot would look fine while
    # being unable to name a single bill. Check it loudly and stop.
    matched = sum(1 for j in jobs if j[3])
    if not jobs:
        sys.exit("\nERROR: no downloaded bill files found under " + pdf_root +
                 "\nRun download_bills.py first.\n")
    if matched == 0:
        sys.exit("\nERROR: no file name matched the metadata. File names must "
                 "look like  0000_370_2019_The Bill.pdf\n"
                 "Check download_bills.py naming and bills_metadata.csv.\n")
    if matched < len(jobs):
        print("WARNING: %d of %d files did not match a metadata row"
              % (len(jobs) - matched, len(jobs)))

    # Count bills by number AND year. A bill number on its own is not a bill:
    # Lok Sabha numbers restart every year, so "Bill No. 82" exists in 22
    # different years and means 22 different laws. Counting the number alone
    # reported 374 "bills" for a corpus that really held 3,643.
    seen_bills = {(j[3].get("billNumber"), j[3].get("billYear"))
                  for j in jobs if j[3]}
    print("found %d downloaded files across %d bills"
          % (len(jobs), len(seen_bills)))
    print("reading and splitting into clauses...\n")

    all_records, quarantined = [], []
    for n, (path, slot, fn, m) in enumerate(jobs, 1):
        if n % 200 == 0 or n == len(jobs):
            print("  %5d / %5d files   %6d chunks so far"
                  % (n, len(jobs), len(all_records)))
        text, why = read_any(path)
        if text is None:
            quarantined.append(quarantine_row(fn, why.split(":")[0], why, m))
            continue
        if not looks_english(text):
            quarantined.append(quarantine_row(fn, "NOT_ENGLISH",
                                              "language check failed", m))
            continue
        try:
            all_records.extend(build_chunks(text, m, slot, fn))
        except Exception as e:
            # One bad file can never kill the run.
            quarantined.append(quarantine_row(fn, "CHUNK_FAILED",
                                              type(e).__name__ + ": "
                                              + str(e)[:120], m))

    print("\nchunks before dedupe: %d" % len(all_records))
    kept = dedupe(all_records)
    print("chunks after dedupe : %d  (%d duplicate copies removed)"
          % (len(kept), len(all_records) - len(kept)))

    # chunk ids must be unique
    used = set()
    for r in kept:
        cid = "%s_%s_%s_%03d" % (r["bill_no"] or "x", r["year"] or "x",
                                 r["slot"][:12] if r["slot"] else "x",
                                 len(used))
        while cid in used:
            cid += "x"
        used.add(cid)
        r["chunk_id"] = cid

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    with open(os.path.join(args.outdir, "chunks.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()) if kept else [])
        w.writeheader()
        for r in kept:
            w.writerow(r)
    with open(os.path.join(args.outdir, "quarantine.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bill_no", "year",
                                          "reason", "detail"])
        w.writeheader()
        for q in quarantined:
            w.writerow(q)

    parts = {}
    for r in kept:
        parts[r["part"]] = parts.get(r["part"], 0) + 1
    print("\nparts: " + ", ".join("%s %d" % kv for kv in sorted(parts.items())))
    print("quarantined files: %d" % len(quarantined))
    reasons = {}
    for q in quarantined:
        reasons[q["reason"]] = reasons.get(q["reason"], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("   %-18s %d" % (k, v))
    print("\nwrote %s/chunks.json, chunks.csv, quarantine.csv" % args.outdir)


if __name__ == "__main__":
    main()
