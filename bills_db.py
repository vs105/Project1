#!/usr/bin/env python3
"""
BILLS FACTS DATABASE (Step 3b) - exact answers for dates, status, counts, lists
================================================================================
RAG (chunk search) is right for "what does clause 5 say?".  It is WRONG for
"when was this bill passed?" or "how many bills were assented in 2019?" -
those are FACTS.  Sansad's metadata already holds every date for every bill,
so we answer facts from a real database: exact, instant, no LLM, no key,
no hallucination.

  INPUT : bills_metadata.csv (written by download_bills.py)
  OUTPUT: answers (text) - or None, meaning "not a facts question, use RAG"

What it can answer (no API key needed for ANY of this):
  * one bill  : introduced / passed LS / passed RS / assent + Act no. /
                committee / ministry / status / type  -> single fact
  * one bill  : "everything / timeline / full details"  -> full profile card
  * many bills: "how many ... in 2019", "list ... between 2015 and 2019"
  * ambiguity : same title in many years -> asks WHICH one (never guesses)
  * data check: flags impossible/odd dates in the source data (shown, not hidden)

Run  : python bills_db.py --ask "When was Bill No. 370 of 2019 passed?"
       python bills_db.py --selftest          # fixed checks + sampled identity/date evaluations
       python bills_db.py --export bills.db   # SQLite file (for dashboard)

VIVA ONE-LINERS (all true, all in this file):
  "Facts come from a database, content comes from RAG - each tool for its job."
  "Ambiguous names get a question back, not a guess."
  "We flag suspicious dates in the official data instead of repeating them."
"""
import argparse
import csv
import difflib
import os
import re
import sqlite3
import threading
import weakref
from attribution import asks_person, recorded_by, conflict_note
from datetime import datetime

# bills_metadata_merge.csv is NOT here: it is the merge REPORT (columns
# action/house/billNumber/billYear/what/title), not metadata. Reading it
# would give every bill an empty title. bills_metadata.csv already holds
# the merged superset, because the RTI merge writes its additions into it.
CSV_CANDIDATES = ["bills_data/bills_metadata.csv", "bills_metadata.csv",
                  "test_data/bills_metadata.csv"]
SRC_LINE = ("_Metadata provenance is not recorded for this dataset copy; "
            "verify the underlying bill record on the official Sansad portal._")

def metadata_source_line(bill):
    primary = str(bill.get("source_primary") or "").strip().lower()
    filled = str(bill.get("filled_from") or "").strip()
    if primary == "sansad":
        note = "_Source: Sansad bill metadata (primary/official)._"
        if filled:
            note += " Some blank fields were supplemented from: " + filled + "."
        return note
    if primary == "rti":
        return "_Source: RTI Wiki metadata (secondary); verify against Sansad._"
    return SRC_LINE

# ------------------------------------------------------------------ identity --
# One bill's identity is (house, bill_number_norm, year). NOT the raw string.
#
# Rajya Sabha numbers are Roman (I, II, ... LXXXVIII) and Lok Sabha numbers are
# Arabic (1..999), so the raw strings never match each other. A user who asks
# about "Bill 38 of 2018" means Rajya Sabha Bill XXXVIII of 2018, and until now
# nothing could connect those two spellings.
#
# Normalising to an integer is what connects them, and it is ALSO why the house
# has to be in the key. Measured on the whole corpus: (norm, year) alone has
# 1,289 collisions, because Lok Sabha Bill 10 of 2004 and Rajya Sabha Bill X of
# 2004 both normalise to 10. (house, norm, year) has 0. Without the house,
# normalising would turn 5,169 unambiguous bills into 1,289 ambiguous ones.

# Only WELL-FORMED Roman numerals. A greedy subtractive reader accepts "IIII"
# as 4 and "IC" as 101, which would silently invent bill numbers that do not
# exist. This pattern rejects both.
_ROMAN_OK = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_VALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
               (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
               (5, "V"), (4, "IV"), (1, "I")]


def roman_to_int(s):
    """'XXXVIII' -> 38. Returns None for anything that is not a well-formed
    Roman numeral, so a junk value can never become a plausible bill number."""
    s = (s or "").strip().upper().replace(" ", "")
    if not s or not _ROMAN_OK.match(s):
        return None
    i = total = 0
    for value, numeral in _ROMAN_VALS:
        while s.startswith(numeral, i):
            total += value
            i += len(numeral)
    return total if i == len(s) else None


def normalize_bill_number(raw):
    """
    'XXXVIII' -> ('XXXVIII', 38)      '38' -> ('38', 38)      '' -> ('', None)

    Returns (cleaned original string, integer or None). The original string is
    kept because it is the OFFICIAL form and must be shown to the user; the
    integer is what matching uses. None means "not a number I understand", and
    callers must fall back to the raw string rather than dropping the bill.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", None
    if re.fullmatch(r"\d{1,4}", raw):
        return raw, int(raw)
    return raw, roman_to_int(raw)


# Which house a question's BILL belongs to. The house word must sit immediately
# in front of the word "bill": "Rajya Sabha Bill No. 38 of 2018" is a Rajya
# Sabha bill. "Bill XXXVIII of 2018 passed by Lok Sabha" is a Rajya Sabha bill
# that the Lok Sabha passed - a different fact - and reading that as a house
# filter answers about the wrong law.
# ... singular OR plural. "how many Rajya Sabha bills in 2004" is the common
# phrasing for a count, and the singular-only pattern matched nothing there.
# One optional word may sit between "Sabha" and "bill(s)" so that
# "Rajya Sabha Government bills" and "Lok Sabha money bills" are filtered
# too. "passed by the Lok Sabha" still cannot match, because there is no
# "bill" word directly after the house name.
HOUSE_BILL = re.compile(r"\b(?:the\s+)?(rajya|lok)\s+sabha\s+(?:\w+\s+)?bills?\b",
                        re.IGNORECASE)


def norm_key(house, bill_no_norm, raw_no, year):
    """
    The unique identity of one bill. house is lowercased because a question
    arrives lowercased and the CSV holds "Lok Sabha".

    When the number will not normalise, the raw string is used instead, so a
    bill is never lost for having an unusual number.
    """
    h = str(house or "").strip().casefold()
    y = str(year or "").strip()
    n = bill_no_norm
    if n is None or n == "":
        n = str(raw_no or "").strip().casefold()
    return (h, str(n), y)


# Bill-title short names people actually type -> words the title search understands.
ALIASES = {
    "caa": "citizenship amendment 2019",
    "dpdp": "digital personal data protection 2023",
    "bns": "bharatiya nyaya sanhita 2023",
    "bnss": "bharatiya nagarik suraksha sanhita 2023",
    "bsa": "bharatiya sakshya 2023",
    "triple talaq": "muslim women protection of rights on marriage 2019",
    "rbi": "reserve bank of india",
    "pocso": "protection of children from sexual offences",
}

TITLE_NOISE = {"the", "bill", "of", "and", "a", "an", "in", "for", "to", "act",
               "on", "by", "with", "etc", "section", "sections"}
QUERY_NOISE = TITLE_NOISE | {
    "when", "was", "were", "is", "are", "what", "which", "who", "whom", "how",
    "many", "did", "does", "do", "get", "got", "its", "it", "this", "that",
    "from", "as", "at", "be", "been", "become", "became", "law", "lok", "sabha",
    "rajya", "passed", "pass", "passes", "introduced", "introduce", "moved",
    "assent", "assented", "date", "dates", "tell", "me", "about", "details",
    "detail", "all", "full", "complete", "timeline", "journey", "history",
    "status", "ministry", "under", "number", "no", "committee", "referred",
    "house", "houses", "both", "give", "show", "explain", "everything",
    "please", "pls", "information", "info", "summary", "progress", "stage",
    "current", "now", "pending", "lapsed", "withdrawn", "president", "acts",
    "bills", "private", "government", "member", "year", "day", "time", "kab",
    "hua", "ka", "ki", "ke", "i", "want", "know", "say", "said", "has",
    "have", "had", "there", "any", "type", "notified", "gazette", "enacted",
    "act", "number", "no", "yes", "then", "also", "and", "or", "whether",
    # ---- question VERBS. These are the words a question adds AROUND a title,
    # and they are the whole bug: the resolver asks that EVERY question token
    # appear in the bill title, so one verb this set did not know about broke
    # title resolution completely. "How was the University of Allahabad Bill,
    # 2004 proposed?" resolved to NOTHING, because "proposed" is not in that
    # title -- the bot then could not answer a question that named its bill in
    # full.
    #
    # ONLY words that appear in ZERO bill titles may go here. Measured against
    # this corpus: "amendment" is in 2,319 titles, "state" in 120, "provision"
    # in 34. Adding any of those makes the resolver too loose and it starts
    # returning the wrong bill -- which is how this list first went wrong.
    "proposed", "propose", "proposal", "proposing", "rejected", "reject",
    "rejecting", "negatived", "negativing", "blocked", "blocking", "defeated",
    "approved", "approve", "tabled", "brought", "mover",
    "ask", "asked", "asking", "mean", "meaning", "definition", "define",
    "defined", "fine", "imprisonment", "clause",
    "says", "stated", "cover", "covers", "covered", "seek", "seeks", "sought",
    "aim", "aims", "purpose", "object", "background",
}
# Words that mean "read the TEXT and tell me what is in it". If any of these
# is in the question it is not a metadata question, no matter which fact words
# also appear. "What does X say about the ministry?" contains "ministry" but is
# asking what the bill SAYS, not which ministry moved it.
# Words that mean "read the TEXT and tell me what is in it". If any of these
# is in the question it is not a metadata question, no matter which fact words
# also appear. "What does X say about the ministry?" contains "ministry" but is
# asking what the bill SAYS, not which ministry moved it.
#
# EVERY entry here was measured against the corpus first, and this list is the
# reason the rule exists: "section" appears in 639 bill titles, "provision" in
# 114, "punishment" in 7, "cover" in 11, "means" in 4, "clause" in 2. Using any
# of those as a marker defers "When was the Delhi Laws (Special Provisions)
# Bill, 2006 passed?" - the title itself carries the marker - and the bot stops
# answering a question it used to answer. Multi-word phrases are safe because
# no title contains them.
#
# A marker is only NEEDED when a question carries both a fact word and a
# content word. Where no fact word matches at all, fact() returns None anyway,
# so single words that are title words were dropped rather than kept.
CONTENT_MARKERS = (
    "say about", "says about", "said about", "state about", "states about",
    "what does it say", "what does the bill say", "what does this say",
    "prescribe", "prescribes", "provide for", "provides for",
    "contain", "contains", "imprison", "explain", "describe", "definition",
    "covers", "objects and reasons", "what is the object", "what happens if",
    "why was", "how does",
)

FACT_WORDS = {  # field -> trigger words/phrases (matched on the lowercase question)
    # "how was it proposed" / "who proposed it" are the same fact as "who
    # introduced it" -- the mover. A user asking how a bill was PROPOSED is
    # asking who moved it, not for the clause text, so these belong here.
    "by": ("who introduced", "introduced by", "mover", "sponsor", "who moved",
           "who proposed", "proposed by", "how was it proposed", "how proposed",
           "who brought", "brought by", "who tabled", "tabled by",
           "who is the author", "moved by"),
    "ministry": ("ministry", "minister", "department"),
    # "become an act" was missing: "When did X become an Act?" fell through to
    # the text search and came back with nothing.
    "assent": ("assent", "became law", "become law", "became an act",
               "become an act", "act number", "act no", "enacted", "notified",
               "gazette", "into law", "received assent", "turned into law",
               "signed into law", "president signed"),
    "committee": ("committee", "referred", "jpc", "select committee"),
    "type": ("government or private", "private member", "type of bill", "what type"),
    # "rejected" was missing, which is the single most natural way to ask
    # whether a bill failed. "Is the X Bill rejected?" fell straight through
    # to the text search. In this data a rejected bill is "Negatived".
    "status": ("status", "pending", "lapsed", "withdrawn", "what happened",
               "current state", "where is", "rejected", "reject", "negatived",
               "negatived", "blocked", "defeated", "thrown out", "killed",
               "did it pass", "did it fail", "was it passed", "was it rejected",
               "did it become law", "approved", "still pending", "in progress"),
}
CARD_WORDS = ("everything", "all details", "full details", "complete details",
              "timeline", "journey", "history", "profile", "progress",
              "overview", "all information", "full information", "all the details",
              "complete information", "all about")
PRONOUNS = {"it", "its", "this", "that", "same", "itself"}
TOKEN = re.compile(r"[a-z0-9]+")

# Month names, so "how many bills passed in January 2020" works.
MONTHS = {}
for _i, _n in enumerate(["january", "february", "march", "april", "may", "june", "july",
                         "august", "september", "october", "november", "december"], 1):
    MONTHS[_n] = _i
    MONTHS[_n[:3]] = _i


def this_period(q, today=None):
    """
    Turn "this year", "last year", "this month", "last month" into a real
    period. Returns (year, month) where either can be None, or None if the
    question names no period at all.

    Without this, "how many bills were released this year" could not be
    answered, because nothing mapped the words "this year" onto 2026.
    """
    q = (q or "").lower()
    if not re.search(r"\b(this|last|past|current)\b", q):
        return None
    import datetime as _dt
    now = today or _dt.date.today()
    year, month = now.year, now.month
    if re.search(r"\blast\b", q) or re.search(r"\bpast\b", q):
        if re.search(r"\b(year)\b", q):
            year -= 1
            month = None
        else:
            month -= 1
            if month == 0:
                month, year = 12, year - 1
    elif not re.search(r"\bmonth\b", q):
        month = None
    return (year, month)
YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")  # FIXED (28 Sep 2026): ANY 4-digit year 1500-2099 is SEEN. Out-of-range years (e.g. 1995, 1860) fall through to RAG/refusal instead of being silently substituted with a different-year bill.


def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def fmt(d):
    return f"{d.day} {d.strftime('%B %Y')}" if d else None


def tokens(s, noise):
    out = []
    for t in TOKEN.findall((s or "").lower()):
        if t in noise or (len(t) == 4 and t.isdigit() and t[:2] in ("19", "20")):
            continue
        out.append(t)
    return out


# Words that carry no meaning when matching a bill name.
NAME_NOISE = {"bill", "bills", "the", "of", "a", "an", "and", "for", "to",
              "on", "in", "with", "amendment"}


def norm_title(name):
    """Squash a bill name down to the words that identify it.

    "The Farmers' Produce Trade and Commerce (Promotion and Facilitation) Bill,
    2020" becomes "farmers produce trade commerce promotion facilitation".
    Case, punctuation, the trailing year and filler words all go, so a name
    typed by a human matches the official uppercase title.
    """
    s = (name or "").lower()
    s = re.sub(r"\b(1[5-9]\d\d|20\d\d)\b", " ", s)      # drop years
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    # A single DIGIT is kept on purpose. "The Appropriation (No.3) Bill" and
    # "(No.2) Bill" differ only by that digit, and dropping it made every
    # Appropriation bill in the corpus normalise to the same key, so the
    # bot asked "which one" about a bill it had been told exactly.
    words = [w for w in s.split()
             if w not in NAME_NOISE and (len(w) > 1 or w.isdigit())]
    return " ".join(words)


def find_csv(path=None):
    for p in ([path] if path else []) + [os.environ.get("BILLS_CSV", "")] + CSV_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


class _ThreadConnection:
    def __init__(self, connection):
        self.connection = connection
        weakref.finalize(self, connection.close)


class BillsDB:
    def __init__(self, csv_path):
        self.csv_path = os.path.abspath(csv_path)
        self.raw_groups = {}
        self.bills = []
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                self.bills.append(self._row(r))
        # THE identity index: (house, bill_number_norm, year). The house is in
        # it because normalising is what makes Lok Sabha Bill 10 and Rajya Sabha
        # Bill X both become 10 - 1,289 such pairs exist in this corpus, so an
        # index without the house would silently return one house's bill for a
        # question about the other's.
        self.by_key = {norm_key(b["house"], b["norm"], b["no"], b["year"]): b
                       for b in self.bills}
        # A bill whose number will not normalise (very rare) still needs a
        # lookup, so its raw lowercased string is indexed too. norm_key already
        # falls back to the raw string, so this is the same key.
        self.by_key_ci = {(b["no"].casefold(), b["year"]): b for b in self.bills}
        # Everything a given (norm, year) points at, split by house. This is
        # what lets the bot say "that number exists in both houses" instead of
        # quietly picking one.
        self.by_numyear = {}
        for b in self.bills:
            self.by_numyear.setdefault((str(b["norm"]), b["year"]), []).append(b)
        for b in self.bills:
            b["toks"] = set(tokens(b["title"], TITLE_NOISE))
            # a bill can be introduced in one year but titled with the next ("... Bill, 2021"
            # numbered 156 of 2019), so a year in the question may match either.
            b["title_years"] = set(YEAR.findall(b["title"].lower()))
        # How fresh this table is, stated as a fact rather than a claim. A bill
        # introduced after the day the metadata was fetched cannot be in it, so
        # the newest introduction date IS the data's as-of date - and the app
        # shows it, instead of letting "covers 2004-2026" imply "up to today".
        intro = [b.get("intro") for b in self.bills]
        intro = [d for d in intro if d]
        self.data_as_of = max(intro).isoformat() if intro else ""
        self.vocab = sorted({t for b in self.bills for t in b["toks"] if len(t) > 3})
        self.ministries = sorted({b["ministry"].lower() for b in self.bills if b["ministry"]},
                                 key=len, reverse=True)
        self.statuses = {b["status"].lower(): b["status"] for b in self.bills if b["status"]}
        for bill in self.bills:
            self.raw_groups.setdefault((bill['no'].casefold(),bill['year']), []).append(bill)
        self._sql_local = threading.local()

    @property
    def db(self):
        # Never hand an SQLite connection created on one Streamlit thread to another.
        if not hasattr(self._sql_local, 'db'):
            self._sql_local.db = _ThreadConnection(self._sqlite())
        return self._sql_local.db.connection

    # ---------- loading ----------
    @staticmethod
    def _row(r):
        g = lambda k: re.sub(r"\s+", " ", (r.get(k) or "").strip())
        docs = {"introduced text": g("billIntroducedFile"), "passed-by-LS text": g("billPassedInLSFile"),
                "passed-by-RS text": g("billPassedInRSFile"),
                "passed-by-both text": g("billPassedInBothHousesFile"),
                "gazette (Act text)": g("billGazettedFile"), "committee report": g("reportFile"),
                "synopsis": g("billSynopsisFile"), "errata": g("errataFile")}
        # "norm" is the integer form of the bill number, written by
        # download_bills.py. "no" keeps the OFFICIAL string, which for a Rajya
        # Sabha bill is Roman and is what the user must be shown.
        raw_no = g("billNumber")
        _raw, norm = normalize_bill_number(raw_no)
        return {"no": raw_no, "norm": norm,
                "year": g("billYear"), "title": g("billName"),
                "type": g("billType"), "category": g("billCategory"),
                "ministry": g("ministryName"),
                # Prefer the canonical "house" column that download_bills.py
                # writes, and fall back to billIntroducedInHouse (present on
                # every legacy row, and what the older index had). The two
                # agree on the current index because repair_legacy() derives
                # one from the other, but they are not the same field: if the
                # canonical column is ever missing or blank, the house - and
                # with it every house-specific resolution - must not go blank.
                "house": g("house") or g("billIntroducedInHouse"),
                "by": g("billIntroducedBy"),
                "source_primary": g("source_primary"),
                "filled_from": g("filled_from"),
                "intro": parse_date(g("billIntroducedDate")),
                "ls": parse_date(g("billPassedInLSDate")), "rs": parse_date(g("billPassedInRSDate")),
                "cmte": parse_date(g("referredToCommitteeDate")),
                "report": parse_date(g("reportPresentedDate")), "assent": parse_date(g("billAssentedDate")),
                "act_no": g("actNo"), "act_year": g("actYear"), "status": g("status"),
                "docs": [k for k, v in docs.items() if v]}

    def _sqlite(self):
        # Queries are still thread-local. Allow the finalizer to close an abandoned
        # thread's private handle even if its owning UI thread has ended.
        db = sqlite3.connect(":memory:", check_same_thread=False)
        # The four *_m columns hold the year and month as "YYYY-MM" so a
        # question like "how many bills were introduced this month" can be
        # answered exactly. Without them only whole years could be counted.
        # `house` is the LAST column on purpose: every query names its columns
        # explicitly, so appending cannot disturb any of them. Without it there
        # is no way to answer "how many Rajya Sabha bills" at all.
        db.execute("CREATE TABLE bills(no TEXT, year TEXT, title TEXT, type TEXT, category TEXT,"
                   " ministry TEXT, intro TEXT, ls TEXT, rs TEXT, assent TEXT, status TEXT,"
                   " intro_m TEXT, ls_m TEXT, rs_m TEXT, assent_m TEXT, house TEXT, number_sort INTEGER)")
        iso = lambda d: d.isoformat() if d else None
        ym = lambda d: d.strftime("%Y-%m") if d else None
        db.executemany("INSERT INTO bills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       [(b["no"], b["year"], b["title"], b["type"], b["category"], b["ministry"],
                         iso(b["intro"]), iso(b["ls"]), iso(b["rs"]), iso(b["assent"]), b["status"],
                         ym(b["intro"]), ym(b["ls"]), ym(b["rs"]), ym(b["assent"]), b["house"],
                         int(b["norm"]) if b["norm"] is not None else 2147483647)
                        for b in self.bills])
        db.commit()
        return db

    def export(self, path):
        if os.path.exists(path):
            os.remove(path)
        out = sqlite3.connect(path)
        self.db.backup(out)
        out.close()

    # ---------- finding ONE bill ----------
    def _fix_typos(self, toks):
        fixed = []
        for t in toks:
            if t in self.vocab or len(t) < 5 or t.isdigit():
                fixed.append(t)
                continue
            m = difflib.get_close_matches(t, self.vocab, n=1, cutoff=0.82)
            fixed.append(m[0] if m else t)
        return fixed

    def resolve(self, question, last_filters=None):
        """-> {'kind': 'one'|'many'|'none', 'bills': [...], 'others': [...]}"""
        q = (question or "").lower()
        for k, v in ALIASES.items():
            q = re.sub(rf"\b{re.escape(k)}\b", v, q)
        years = YEAR.findall(q)
        mh = HOUSE_BILL.search(q)
        want_house = (("rajya sabha" if mh.group(1) == "rajya" else "lok sabha") if mh else None)
        # "Bill No. 370 of 2019" / "bill 370/2019" / "370/2019". Needs the word "bill" BEFORE
        # the number, so a title like "Appropriation (No.3) Bill" is not misread as Bill No. 3.
        # A bill number is DIGITS (Lok Sabha 1..999) or a ROMAN NUMERAL
        # (Rajya Sabha I..LXXXVIII, plus the Lok Sabha C-series). The old
        # pattern was (\d{1,3}) only, so "Bill No. I of 2004" matched NOTHING
        # and fell through to the title search, which cannot find it either.
        # The bot could not identify any of the 1,491 Rajya Sabha bills by
        # number - the single worst gap in the whole system, because the
        # Rajya Sabha is 29% of the corpus.
        m = re.search(r"\bbill\s*(?:no\.?|number|num\.?)?\s*(\d{1,3})\b"
                      r"(?:\s*(?:of|/|,|\(|-)\s*(20[0-2]\d))?", q)
        if not m:
            # A bare Roman numeral is only read as a bill number when the
            # YEAR is there too, or when "no." said so explicitly above.
            # Without that guard, "the bill I introduced yesterday" would read
            # "I" as Bill No. I and answer about the wrong law entirely.
            # "no."/"number" is optional here too, because the common phrasing
            # is "Bill No. I of 2004" - with the words in it.
            m = re.search(r"\bbill\s*(?:no\.?|number|num\.?)?\s*"
                          r"([ivxlcdm]{1,8})\s*(?:of|/|,)\s*(20[0-2]\d)\b", q)
        if not m:
            m = re.search(r"\b([ivxlcdm]{1,8})\s*/\s*(20[0-2]\d)\b", q)
        if m:
            no = m.group(1).upper() if m.group(1).isalpha() else m.group(1)
            yr = (m.group(2) if m.lastindex and m.lastindex >= 2 else None)
            yr = yr or (years[-1] if years else None)
            if yr:
                want_house = None
                mh = HOUSE_BILL.search(q)
                if mh:
                    want_house = "rajya sabha" if mh.group(1) == "rajya" \
                        else "lok sabha"
                _raw, nrm = normalize_bill_number(no)

                # (1) EXACT official number first, but ONLY when the question
                #     spelled the number in a non-Arabic form. "Bill XXXVIII of
                #     2018" is the Rajya Sabha bill's own number spelled the way
                #     Parliament spells it, so an exact match is the strongest
                #     signal there is - it is not a guess. This is also what
                #     keeps an explicitly Roman question resolving to one bill:
                #     without it, XXXVIII normalises to 38 and collides with Lok
                #     Sabha Bill 38 of the same year.
                #     The same is true of an ARABIC number, and this is measured,
                #     not assumed: of the 1,289 (number, year) pairs that BOTH
                #     houses hold, the Arabic spelling is the Lok Sabha bill in
                #     1,289 of them and the Roman spelling the Rajya Sabha bill
                #     in 1,289 of them - never the other way round. So the form
                #     the user typed is a complete disambiguator, and matching it
                #     exactly is evidence, not a coin toss.
                # The house named in the question outranks everything: asking
                # about "Rajya Sabha Bill 38" and being shown the Lok Sabha
                # bill is answering about the wrong law, however exact the
                # number match was.
                exact = [b for b in self.bills
                         if b["year"] == yr
                         and str(b["no"]).strip().casefold() == str(no).casefold()
                         and (want_house is None
                              or str(b["house"] or "").strip().casefold() == want_house)]
                if len(exact) == 1:
                    # The spelling decides when both houses hold the number, but
                    # the TITLE must still be able to override it: a question
                    # naming the other house's bill by its title, with this
                    # number, is a real ambiguity, not a request for this bill.
                    if want_house is None:
                        group = self.by_numyear.get((str(nrm), yr), [])
                        if len({g["house"] for g in group}) > 1:
                            words = set(tokens(q, QUERY_NOISE))
                            others = [g for g in group if g is not exact[0]]
                            chosen = exact[0]
                            best = max(others, key=lambda g: len(words & g["toks"]))
                            mine = len(words & chosen["toks"])
                            theirs = len(words & best["toks"])
                            # One shared word ("Bill") is not evidence; a title
                            # that fits the other house's bill far better is.
                            if theirs >= 2 and theirs >= mine + 2:
                                return {"kind": "many", "bills": [chosen, best],
                                        "others": [], "why": "both houses"}
                    return {"kind": "one", "bills": exact, "others": []}
                if len(exact) > 1:
                    return {"kind": "many", "bills": exact, "others": [],
                            "why": "both houses"}

                # (2) NORMALISED. This is the bridge: "Bill 38 of 2018" can now
                #     reach Rajya Sabha Bill XXXVIII of 2018, which used to be
                #     unreachable because the strings never matched.
                hits = [b for b in self.by_numyear.get((str(nrm), yr), [])
                        if want_house is None
                        or str(b["house"] or "").strip().casefold() == want_house]
                if len(hits) == 1:
                    return {"kind": "one", "bills": hits, "others": []}
                if len(hits) > 1:
                    # Same number, same year, BOTH houses. Answering would mean
                    # answering about the wrong law, so the caller is told and
                    # shows both. 1,289 such pairs exist in this corpus.
                    return {"kind": "many", "bills": hits, "others": [],
                            "why": "both houses"}

                # (3) A number that will not normalise at all.
                b = self.by_key_ci.get((no.casefold(), yr))
                if b and want_house and b['house'].strip().casefold() != want_house:
                    b = None
                return {"kind": "one", "bills": [b], "others": []} if b else \
                    {"kind": "none", "bills": [], "others": [], "why": f"no Bill No. {no} of {yr}"}
            same = [b for b in self.bills if b['no'].casefold() == no.casefold()
                    and (not want_house or b['house'].strip().casefold() == want_house)]
            return {"kind": "many", "bills": same, "others": []} if same else \
                {"kind": "none", "bills": [], "others": []}
        # A question that only points back - "this bill", "it", "the same
        # bill" - is answered by the bill the previous question resolved to.
        #
        # This runs BEFORE the title search on purpose. It used to run after,
        # so "what are all clauses for this bill?" was first matched against
        # every bill whose title contains "clauses" - the two General Clauses
        # (Amendment) Bills - and answered about the wrong law. Muting
        # "clauses"/"general" in QUERY_NOISE would have fixed that one example
        # and broken real title questions, so the ORDER is what changed.
        #
        # It is safe because a question that names its own bill also names a
        # year, a house or a bill number, and any of those disables this path.
        pronoun_points_back = (
            set(re.findall(r"[a-z]+", q)) & PRONOUNS
            and not years and not want_house
            and not re.search(r"\bbill\s*(?:no\.?|number|num\.?)?\s*[\divxlcdm]{1,8}\b", q))
        if pronoun_points_back:
            if last_filters and last_filters[0] and last_filters[1]:
                # last_filters is (year, no, house). Build the key exactly the
                # way by_key was built: norm_key(house, norm, year), a 3-tuple.
                # An older lookup used by_key.get((no, year)) - a 2-tuple
                # against a 3-tuple index - so it could never match.
                _raw, nrm = normalize_bill_number(last_filters[1])
                prev_house = last_filters[2] if len(last_filters) > 2 else None
                b = self.by_key.get(norm_key(prev_house, nrm, last_filters[1], last_filters[0]))
                if b is None and not prev_house:      # never cross an explicit House
                    b = self.by_key_ci.get((str(last_filters[1]).casefold(),
                                            str(last_filters[0])))
                if b:
                    return {"kind": "one", "bills": [b], "others": []}
            # A pointing-back question with NO bill in hand must not fall
            # through to the title search either: "what are all clauses for
            # this bill?" would otherwise list the General Clauses (Amendment)
            # Bills - a confident answer about the wrong law. Measured safe:
            # no real title contains it/its/this/that/same/itself as a whole
            # word (0 of the 5,169 titles do), so nothing legitimate is lost
            # by stopping here.
            return {"kind": "none", "bills": [], "others": [],
                    "why": "the question points back at a bill, but no bill has been named yet"}

        qt = self._fix_typos(tokens(q, QUERY_NOISE))
        if qt:
            pool = [b for b in self.bills
                    if (not years or b["year"] == years[-1] or years[-1] in b["title_years"])
                    and (not want_house or b['house'].strip().casefold() == want_house)]
            qs = set(qt)
            cand = [b for b in pool if qs <= b["toks"]]
            if cand:
                cand.sort(key=lambda b: -len(qs) / len(qs | b["toks"]))
                top = len(qs) / len(qs | cand[0]["toks"])
                second = len(qs) / len(qs | cand[1]["toks"]) if len(cand) > 1 else 0
                if len(cand) == 1 or top - second >= 0.15:
                    return {"kind": "one", "bills": [cand[0]], "others": cand[1:4]}
                return {"kind": "many", "bills": cand, "others": []}
            # A CONTENT question carries words the title never had. "What
            # punishment does the BNS prescribe for murder?" asks about murder,
            # and no bill is titled Murder - so the subset test above fails on
            # every content question and the bill cannot be named at all. That
            # is why those questions came back "I could not find a bill in my
            # data". Score the other way round instead: how much of the TITLE
            # does the question cover? "bharatiya nyaya sanhita" sits whole
            # inside that question, which is enough to name the bill, and the
            # question's own words are simply not part of the test.
            #
            # Two guards, because coverage alone is a loose test. The title must
            # hold at least two words and at least two of them must be present,
            # so a one-word title cannot be claimed by a single stray word.
            scored = sorted(
                ((len(qs & b["toks"]) / len(b["toks"]), b)
                 for b in pool
                 if len(b["toks"]) >= 2 and len(qs & b["toks"]) >= 2),
                key=lambda x: -x[0])
            scored = [(c, b) for c, b in scored if c >= 0.8]
            if scored:
                top = scored[0][0]
                same = [b for c, b in scored if c >= top - 0.15]
                if len(same) == 1:
                    return {"kind": "one", "bills": [same[0]],
                            "others": [b for _c, b in scored[1:4]]}
                # Two bills share the title - 173 and 121 of 2023 both are the
                # Bharatiya Nyaya Sanhita. Say so and let the caller pick.
                return {"kind": "many", "bills": same, "others": []}
        return {"kind": "none", "bills": [], "others": []}

    # ---------- answering ONE bill ----------
    @staticmethod
    def flags(b):
        f = []
        if b["intro"] and b["ls"] and b["ls"] < b["intro"]:
            f.append("Lok Sabha passing date is BEFORE the introduction date")
        if b["intro"] and b["ls"] and b["ls"].year > b["intro"].year and \
                (b["ls"].month, b["ls"].day) == (b["intro"].month, b["intro"].day):
            f.append(f"Lok Sabha passing date is exactly {b['ls'].year - b['intro'].year} year(s) after "
                     "introduction on the same day/month - possible year typo in the source")
        if b["house"] == "Lok Sabha" and b["ls"] and b["rs"] and b["rs"] < b["ls"]:
            f.append("Rajya Sabha date is before Lok Sabha date for a Lok Sabha bill")
        last = max([d for d in (b["ls"], b["rs"]) if d], default=None)
        if b["assent"] and last and b["assent"] < last:
            f.append("assent date is before the passing date")
        if b["status"] == "Assented" and not b["assent"]:
            f.append("status says Assented but no assent date is recorded")
        return f

    def _head(self, b):
        # The house is in the heading because the same bill number means
        # different laws in the two houses, and "Bill No. I of 2004" looks
        # identical whether it came from the Lok Sabha or the Rajya Sabha. A
        # reader who is checking an answer needs to see which one it was.
        return (f"**{b['title']}** (Bill No. {b['no']} of {b['year']}"
                + (f", {b['house']}" if b["house"] else "") + ")")

    def card(self, b):
        n = lambda d, empty="not recorded": fmt(d) or empty
        act = f" - Act No. {b['act_no']} of {b['act_year']}" if b["act_no"] else ""
        lines = [f"### {b['title']}", f"Bill No. {b['no']} of {b['year']} · {b['type'] or '?'} · "
                 f"{b['category'] or '?'} · Ministry: {b['ministry'] or 'not recorded'}",
                 "", "**Timeline**",
                 f"- Introduced ({b['house'] or 'house not recorded'}): {n(b['intro'])}"
                 + (f" — {conflict_note(b)}" if conflict_note(b) else (f" (metadata: {b['by']})" if b['by'] else "")),
                 f"- Referred to committee: {n(b['cmte'], 'not referred / not recorded')}",
                 f"- Committee report presented: {n(b['report'], 'none recorded')}",
                 f"- Passed by Lok Sabha: {n(b['ls'])}", f"- Passed by Rajya Sabha: {n(b['rs'])}",
                 f"- Presidential assent: {n(b['assent'])}{act}", "",
                 f"**Status:** {b['status'] or 'not recorded'}",
                 "**Official documents on Sansad:** " + (", ".join(b["docs"]) or "none listed")]
        for w in self.flags(b):
            lines.append(f"\n⚠️ Data check: {w} - worth verifying on sansad.in.")
        return "\n".join(lines)

    def fact(self, b, q):
        """Lines for exactly the fields asked. None if the question isn't a fact question."""
        # A question about the TEXT is never a metadata question, however many
        # fact words it happens to contain. Without this guard, "What does Bill
        # 370 of 2019 say about the ministry?" was answered "Ministry: HOME
        # AFFAIRS" - a real answer to a question that was not asked. Checked
        # first, so it beats every FACT_WORDS match below.
        if any(w in q for w in CONTENT_MARKERS):
            return None
        has = lambda words: any(w in q for w in words)
        want = [k for k, w in FACT_WORDS.items() if has(w)]
        if asks_person(q) and 'by' not in want:
            want.insert(0, 'by')
        passed = bool(re.search(r"\bpass(ed|es)?\b", q))
        if "rajya" in q and passed:
            want.append("rs")
        if "lok sabha" in q and passed:
            want.append("ls")
        if passed and "rs" not in want and "ls" not in want:
            # Report the bill's OWN house first. Leading a Rajya Sabha bill
            # with "No Lok Sabha passing date is recorded" reads as "it was
            # never passed", when the truth is simply that this bill belongs
            # to the other house. Order is what fixes that, not wording.
            own = "rs" if "rajya" in (b["house"] or "").lower() else "ls"
            want += [own] + [h for h in ("ls", "rs") if h != own]
        # "How was the X Bill proposed?" asks the same fact as "who moved it".
        # A substring check cannot see it, because the words sit either side
        # of the bill's name, so this is a pattern rather than a phrase.
        if (re.search(r"\bhow\s+(was|were|is|did)\b[^?]*\bpropos", q)
                or re.search(r"\bwho\s+\w*\s*propos", q)) and "by" not in want:
            want.append("by")
        if re.search(r"\b(introduc\w*|moved|brought|tabled)\b", q) and "by" not in want:
            want.append("intro")
        out = []
        for k in dict.fromkeys(want):
            if k == "intro":
                out.append(f"Introduced in {b['house'] or 'the house'} on {fmt(b['intro'])}." if b["intro"]
                           else "No introduction date is recorded.")
            elif k == "ls":
                if b["ls"]:
                    out.append(f"Passed by Lok Sabha on {fmt(b['ls'])}.")
                else:
                    out.append(f"No Lok Sabha passing date is recorded (status: "
                               f"{b['status'] or 'unknown'}).")
            elif k == "rs":
                if b["rs"]:
                    out.append(f"Passed by Rajya Sabha on {fmt(b['rs'])}.")
                else:
                    out.append("No Rajya Sabha passing date is recorded.")
            elif k == "assent":
                out.append(f"Presidential assent on {fmt(b['assent'])}"
                           + (f" (Act No. {b['act_no']} of {b['act_year']})." if b["act_no"] else ".")
                           if b["assent"] else "No assent date is recorded, so it did not become an Act "
                           f"in this data (status: {b['status'] or 'unknown'}).")
            elif k == "committee":
                out.append((f"Referred to a committee on {fmt(b['cmte'])}." if b["cmte"] else
                            "No committee referral is recorded.")
                           + (f" Report presented on {fmt(b['report'])}." if b["report"] else ""))
            elif k == "ministry":
                out.append(f"Ministry: {b['ministry']}." if b["ministry"] else "No ministry is recorded "
                           "(common for private member bills).")
            elif k == "by":
                out.append(recorded_by(b))
            elif k == "type":
                out.append(f"It is a {b['type']} bill ({b['category'] or 'category not recorded'}).")
            elif k == "status":
                out.append(f"Status: {b['status'] or 'not recorded'}.")
        return out or None

    def one_bill(self, res, q):
        b = res["bills"][0]
        if any(w in q for w in CARD_WORDS):
            body = self.card(b)
        else:
            lines = self.fact(b, q)
            if not lines:
                return None
            body = self._head(b) + ": " + " ".join(lines)
            for w in self.flags(b):
                body += f"\n\n⚠️ Data check: {w} - worth verifying on sansad.in."
        if res["others"]:
            body += "\n\nOther close matches: " + "; ".join(
                f"{o['title']} (No. {o['no']}/{o['year']})" for o in res["others"][:3]) + \
                " - name the bill number if you meant another."
        return body + "\n\n" + metadata_source_line(b)

    def many(self, res, question=""):
        bl = res["bills"]
        mh = re.search(r"\b(lok|rajya)\s+sabha\b", question or "", re.I)
        if mh:
            wanted = mh.group(1).casefold() + " sabha"
            if not any(b["house"].strip().casefold() == wanted for b in bl):
                return (f"No {'Lok' if wanted.startswith('lok') else 'Rajya'} Sabha bill "
                        f"numbered {bl[0]['no']} of {bl[0]['year']} is in this data, so I will "
                        "not substitute another bill. The same number/year exists in: "
                        + ", ".join(sorted({b['house'] for b in bl})) + ".")
        lines = [f"- {b['title']} - Bill No. {b['no']} of {b['year']} ({b['type']}, {b['status'] or '?'})"
                 for b in bl[:8]]
        more = f"\n...and {len(bl) - 8} more." if len(bl) > 8 else ""
        head = f"I found {len(bl)} bills that match, so I won't guess:\n"
        # Two different houses can hold the SAME number in the same year - the
        # Rajya Sabha numbers its bills I, II, III ... and the Lok Sabha
        # numbers its own 1, 2, 3 ..., so "Bill No. 38 of 2018" is really two
        # different laws. Telling the user to supply a year or a number they
        # already gave is a dead end; tell them the one thing that separates
        # the two.
        if res.get("why") == "both houses":
            houses = " and ".join(dict.fromkeys(
                b["house"] for b in bl if b.get("house")))
            tail = (f"\n\nBoth the {houses} have a bill with that number for that "
                    f"year. Tell me which one you mean (e.g. \"Rajya Sabha Bill No. "
                    f"{bl[-1]['no']} of {bl[-1]['year']}\").")
        else:
            # Ask for the ONE thing that actually separates the bills shown. If
            # they all share a year - "When was BNS passed?" is two different
            # Bharatiya Nyaya Sanhita bills, both of 2023 - then asking for the
            # year is a dead end: the user already gave it. Ask for the number
            # instead, which is what the list above just showed them.
            years = {b["year"] for b in bl}
            if len(years) == 1:
                tail = ("\n\nThese are all from " + bl[0]["year"] + ". Tell me "
                        "the bill number (e.g. \"Bill No. "
                        + bl[0]["no"] + " of " + bl[0]["year"] + "\").")
            else:
                tail = ("\n\nTell me the year or the bill number (e.g. "
                        "\"Bill No. 370 of 2019\").")
        return head + "\n".join(lines) + more + tail

    # ---------- MANY bills: counts and lists ----------
    def only_metadata_words(self, q):
        """True when every word of the question is a metadata word.

        Metadata words are: a word that appears in some bill title, a number,
        or one of the trigger words this file already looks for (pass, assent,
        private, government, list, how many, the status words, and so on).
        Anything else means the question is asking about bill CONTENT, which
        the metadata cannot answer.
        """
        known = set()
        for words in FACT_WORDS.values():
            known.update(w for phrase in words for w in TOKEN.findall(phrase))
        known.update(w for w in TOKEN.findall(" ".join(CARD_WORDS)))
        known.update(TOKEN.findall(
            "list show name which how many count number of bills bill between "
            "from and to the of in on for by with assent became become law "
            "enacted into notified gazette received passed pass passing "
            "introduced introduc private member government status pending "
            "lapsed withdrawn year years month months rajya sabha lok sabha "
            "total all any january february march april may june july "
            "august september october november december jan feb mar apr "
            "jun jul aug sep sept oct nov dec "
            "were was is are did do does what when where who whom tell me give "
            "please about under into over after before during same it its this "
            "that these those there here more less than only just"))
        for b in self.bills:
            known.update(TOKEN.findall(b["title"].lower()))
        long_known = sorted((w for w in known if len(w) >= 4), key=len, reverse=True)
        for word in TOKEN.findall(q):
            if word in known or word.isdigit():
                continue
            # crude stemming: "assented" and "assenting" are the word "assent"
            if any(word.startswith(k) for k in long_known):
                continue
            return False
        return True

    def metadata_source_summary(self):
        """Describe the provenance recorded in this metadata CSV, without assuming it."""
        counts = {"sansad": 0, "rti": 0, "unknown": 0}
        for bill in self.bills:
            source = str(bill.get("source_primary") or "").strip().lower()
            if source in ("sansad", "rti"):
                counts[source] += 1
            else:
                counts["unknown"] += 1
        parts = []
        if counts["sansad"]:
            parts.append(f"{counts['sansad']} Sansad-primary records")
        if counts["rti"]:
            parts.append(f"{counts['rti']} RTI Wiki-primary records (secondary source)")
        if counts["unknown"]:
            parts.append(f"{counts['unknown']} records with source provenance not recorded")
        return "_Metadata provenance: " + "; ".join(parts) + ". Verify bill details against Sansad._"

    def aggregate(self, question):
        q = question.lower()
        toks = set(TOKEN.findall(q))
        if "bills" not in toks or toks & {"clause", "clauses", "section", "sections", "pages", "words", "chapter"}:
            return None
        counting = "how many" in q or "count" in q or "number of" in q
        listing = bool(re.search(r"\b(list|show|name|which)\b", q))
        if not (counting or listing):
            return None
        # A listing/counting question can only be answered from metadata when
        # every word of it is a metadata word. "Which 2023 bills replaced
        # colonial-era criminal laws?" used to be answered with a plain list of
        # all 98 bills of 2023, which ignores what was actually asked. If a
        # word appears in no bill title and is not a known metadata word, the
        # question is about the CONTENT of the bills, so let RAG handle it.
        if not self.only_metadata_words(q):
            return None
        where, args, said = ["1=1"], [], []
        metric = "year"  # introduced year by default
        # "became Acts" / "became law" / "enacted" / "into law" all mean the
        # same thing here: count by ASSENT date. "became acts" was missing, so
        # "How many bills of 2004 became Acts?" fell through to the default
        # year metric and answered 103 - the total - instead of 31.
        if re.search(r"assent|became?\s+(?:law|acts?)|become\s+(?:law|acts?)"
                     r"|enacted|into\s+(?:law|acts?)", q):
            metric, said = "assent", ["by assent date"]
        elif "rajya sabha" in q and re.search(r"pass", q):
            metric, said = "rs", ["by Rajya Sabha passing date"]
        elif re.search(r"\bpass", q):
            metric, said = "ls", ["by Lok Sabha passing date"]
        col = {"year": "year", "assent": "substr(assent,1,4)", "rs": "substr(rs,1,4)",
               "ls": "substr(ls,1,4)"}[metric]
        mcol = {"year": "intro_m", "assent": "assent_m", "rs": "rs_m",
                "ls": "ls_m"}[metric]
        if metric != "year":
            where.append({"assent": "assent", "rs": "rs", "ls": "ls"}[metric] + " IS NOT NULL")

                # A year named in the question is read FIRST, so that "this year" can
        # only fill in a year when the question does not already name one.
        # Getting this order wrong made "this year" return all 3,678 bills.
        yrs = [int(y) for y in YEAR.findall(q)]
        rng = re.search(r"(?:between|from)?\s*(20[0-2]\d)\s*(?:and|to|-)\s*(20[0-2]\d)", q)

        # A month, either named ("January 2020") or implied ("this month").
        mon = None
        for name, num in MONTHS.items():
            if re.search(rf"\b{name}\b", q):
                mon = num
                said.append(f"month {num:02d}")
                break
        period = this_period(question)
        if period:
            period_year, period_month = period
            if not yrs and not rng:
                yrs = [period_year]
                said.append("this year" if period_month is None else "this month")
            if period_month and not mon:
                mon = period_month

        # apply the year filter
        if rng:
            lo, hi = sorted((int(rng.group(1)), int(rng.group(2))))
            where.append(f"CAST({col} AS INTEGER) BETWEEN {lo} AND {hi}")
            said.append(f"{lo}-{hi}")
        elif yrs:
            where.append(f"CAST({col} AS INTEGER) = {yrs[0]}")
            said.append(str(yrs[0]))

        # apply the month filter
        if mon:
            where.append(f"{mcol} = ?")
            args.append("%s-%02d" % (yrs[0] if yrs else "%", mon))
            if not any(s.startswith("month ") for s in said):
                said.append(f"month {mon:02d}")
        if re.search(r"\bprivate\b", q):
            where.append("type = 'Private Member'")
            said.append("private member bills")
        elif "government" in q:
            where.append("type = 'Government'")
            said.append("government bills")
        if metric == "year":
            for word, real in self.statuses.items():
                if re.search(rf"\b{re.escape(word)}\b", q):
                    where.append("status = ?")
                    args.append(real)
                    said.append(f"status {real}")
                    break
        for mn in self.ministries:
            if re.search(rf"\b{re.escape(mn)}\b", q):
                where.append("lower(ministry) = ?")
                args.append(mn)
                said.append(f"ministry {mn.title()}")
                break
        if "money bill" in q:
            where.append("category LIKE '%Money%'")
            said.append("money bills")
        # "Rajya Sabha bills" names the house the bill came FROM. "bills passed
        # by the Rajya Sabha" names the house that PASSED it - a different
        # question, and one a Lok Sabha bill routinely satisfies, so it is not
        # narrowed here. Same rule as the resolver: the house only counts when
        # it sits directly in front of the word "bill". Without this,
        # "how many Rajya Sabha bills in 2004" answered with the total for the
        # year, both houses together.
        mh = HOUSE_BILL.search(q)
        if mh:
            want = "Rajya Sabha" if mh.group(1) == "rajya" else "Lok Sabha"
            where.append("house = ?")
            args.append(want)
            said.append("%s bills" % want)
        w = " AND ".join(where)
        total = self.db.execute(f"SELECT COUNT(*) FROM bills WHERE {w}", args).fetchone()[0]
        basis = "counted by introduction year" if metric == "year" else said[0]
        crit = "; ".join(s for s in said if s != said[0] or metric == "year") or "all bills"
        if counting:
            out = f"**{total} bills** match ({crit}; {basis})."
            # Say how many bills the corpus has for this measure at all. A count
            # of "40 bills passed in 2023" is misleading if it is 40 out of the
            # 771 bills that have a passing date, and the bot must not imply it
            # counted all 3,678.
            if metric != "year":
                dcol = {"assent": "assent", "rs": "rs", "ls": "ls"}[metric]
                have = self.db.execute(
                    f"SELECT COUNT(*) FROM bills WHERE {dcol} IS NOT NULL").fetchone()[0]
                out += (f"\n\nThat counts only the **{have}** bills in the corpus that "
                        f"have a recorded {dcol.upper()} date at all.")
            if rng and total:
                rows = self.db.execute(f"SELECT {col}, COUNT(*) FROM bills WHERE {w} "
                                       "GROUP BY 1 ORDER BY 1", args).fetchall()
                out += "\n\nBy year: " + ", ".join(f"{y}: {c}" for y, c in rows)
        else:
            rows = self.db.execute(f"SELECT title, no, year, status, house FROM bills WHERE {w} "
                                   "ORDER BY year, house, number_sort, no LIMIT 12", args).fetchall()
            out = f"**{total} bills** match ({crit}; {basis})." + \
                  ("\n\n" + "\n".join(f"- {t} - Bill No. {n} of {y} ({h or 'house not recorded'}, {s})" for t, n, y, s, h in rows)
                   if rows else "")
            if total > 12:
                out += f"\n...showing 12 of {total}. Add a year, type or ministry to narrow it."
        return out + "\n\n" + self.metadata_source_summary()

    # ---------- name to bill number (the resolver) ----------
    def find_by_title(self, name, year=None):
        """
        Look up a bill by its NAME and return the matching bills.

        This is what lets someone type "The Farmers' Produce Trade and Commerce
        (Promotion and Facilitation) Bill" with no year and still land on Bill
        113 of 2020. It is also what lets the bot say plainly that no bill of
        that name exists, instead of quietly answering from a different bill.

        Returns a list, usually empty, one bill, or a handful when a short name
        like "the Finance Bill" is used across many years. The caller decides:
        one match means lock onto it, several means ask which year, none means
        say the bill is not in the corpus.

        It does NOT understand nicknames. "Triple Talaq" and "CAA" appear in no
        official title at all. Those are handled by the caller asking the model
        to propose the official title, which is then checked here. That way a
        guess can never be mistaken for a fact.
        """
        q = norm_title(name)
        if len(q) < 4:
            return []
        qt = set(q.split())
        exact, subset, contains = [], [], []
        for b in self.bills:
            if year and b["year"] != str(year):
                continue
            t = norm_title(b["title"])
            if not t:
                continue
            if t == q:
                exact.append(b)
            elif len(qt) >= 2 and qt <= set(t.split()):
                subset.append(b)
            elif q in t:
                contains.append(b)
        for group in (exact, subset, contains):
            if group:
                # Government bills first, then the shortest title that still
                # matched. A Government bill is the one that becomes law, so
                # when a name is shared by both it is the one a question is
                # about. Of the 3,678 bills here, 932 are Government and 2,746
                # are Private Member's, and almost none of those ever pass.
                return sorted(group, key=lambda b: (
                    not str(b.get("type") or "").lower().startswith("gov"),
                    len(b["title"])))
        return []

    # ---------- the ONE entry point ----------
    def answer(self, question, last_filters=None):
        """-> (reply, (year, bill_no) or None) . reply None = not a facts question."""
        q = (question or "").lower()
        explicit = re.search(r"(bill\s*(no|number)\b|\b\d{1,3}\s*/\s*20[0-2]\d\b)", q)
        if not explicit:
            agg = self.aggregate(question)
            if agg:
                return agg, None
        res = self.resolve(question, last_filters)
        if res["kind"] == "one":
            b = res["bills"][0]
            txt = self.one_bill(res, q)
            return (txt, (b["year"], b["no"])) if txt else (None, (b["year"], b["no"]))
        if res["kind"] == "many":
            has_fact = any(w in q for ws in FACT_WORDS.values() for w in ws) or \
                any(w in q for w in CARD_WORDS) or re.search(r"\b(pass|introduc)\w*", q)
            return (self.many(res), None) if has_fact else (None, None)
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Facts database for Lok Sabha bills")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ask")
    ap.add_argument("--export")
    ap.add_argument("--find-title", default="",
                    help="look up a bill by name and print every match")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    path = find_csv(a.csv)
    if not path:
        raise SystemExit("bills_metadata.csv not found - run download_bills.py first (or pass --csv).")
    db = BillsDB(path)
    print(f"loaded {len(db.bills)} bills from {path}")
    if a.export:
        db.export(a.export)
        print("wrote", a.export)
    if a.find_title:
        hits = db.find_by_title(a.find_title)
        if not hits:
            print("NO MATCH - no bill of that name is in the corpus")
        else:
            print(f"{len(hits)} match(es):")
            for b in hits[:15]:
                print(f"  Bill No. {b['no']} of {b['year']} | {b['type']} | "
                      f"{b['status']} | {b['title']}")
    if a.ask:
        print(db.answer(a.ask)[0] or "(not a facts question -> would go to RAG)")
    if a.selftest:
        import bills_db_selftest
        failures = bills_db_selftest.run(db)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
