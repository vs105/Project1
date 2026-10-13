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
       python bills_db.py --selftest          # 20 built-in checks
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
from datetime import datetime

CSV_CANDIDATES = ["bills_data/bills_metadata.csv", "bills_metadata.csv",
                  "test_data/bills_metadata.csv"]
SRC_LINE = "_Source: Sansad bill metadata (official)._"

# Bill-title short names people actually type -> words the title search understands.
ALIASES = {
    "caa": "citizenship amendment 2019",
    "dpdp": "digital personal data protection 2023",
    "bns": "bharatiya nyaya sanhita 2023",
    "bnss": "bharatiya nagarik suraksha sanhita 2023",
    "bsa": "bharatiya sakshya 2023",
    "triple talaq": "muslim women protection of rights on marriage 2019",
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
}
FACT_WORDS = {  # field -> trigger words/phrases (matched on the lowercase question)
    "by": ("who introduced", "introduced by", "mover", "sponsor", "who moved"),
    "ministry": ("ministry", "minister", "department"),
    "assent": ("assent", "became law", "become law", "act number", "act no",
               "enacted", "notified", "gazette", "into law", "received assent"),
    "committee": ("committee", "referred", "jpc", "select committee"),
    "type": ("government or private", "private member", "type of bill", "what type"),
    "status": ("status", "pending", "lapsed", "withdrawn", "what happened",
               "current state", "where is"),
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
    words = [w for w in s.split() if w not in NAME_NOISE and len(w) > 1]
    return " ".join(words)


def find_csv(path=None):
    for p in ([path] if path else []) + [os.environ.get("BILLS_CSV", "")] + CSV_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


class BillsDB:
    def __init__(self, csv_path):
        self.bills = []
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                self.bills.append(self._row(r))
        self.by_key = {(b["no"], b["year"]): b for b in self.bills}
        for b in self.bills:
            b["toks"] = set(tokens(b["title"], TITLE_NOISE))
            # a bill can be introduced in one year but titled with the next ("... Bill, 2021"
            # numbered 156 of 2019), so a year in the question may match either.
            b["title_years"] = set(YEAR.findall(b["title"].lower()))
        self.vocab = sorted({t for b in self.bills for t in b["toks"] if len(t) > 3})
        self.ministries = sorted({b["ministry"].lower() for b in self.bills if b["ministry"]},
                                 key=len, reverse=True)
        self.statuses = {b["status"].lower(): b["status"] for b in self.bills if b["status"]}
        self.db = self._sqlite()

    # ---------- loading ----------
    @staticmethod
    def _row(r):
        g = lambda k: re.sub(r"\s+", " ", (r.get(k) or "").strip())
        docs = {"introduced text": g("billIntroducedFile"), "passed-by-LS text": g("billPassedInLSFile"),
                "passed-by-RS text": g("billPassedInRSFile"),
                "passed-by-both text": g("billPassedInBothHousesFile"),
                "gazette (Act text)": g("billGazettedFile"), "committee report": g("reportFile"),
                "synopsis": g("billSynopsisFile"), "errata": g("errataFile")}
        return {"no": g("billNumber"), "year": g("billYear"), "title": g("billName"),
                "type": g("billType"), "category": g("billCategory"),
                "ministry": g("ministryName"), "house": g("billIntroducedInHouse"),
                "by": g("billIntroducedBy"), "intro": parse_date(g("billIntroducedDate")),
                "ls": parse_date(g("billPassedInLSDate")), "rs": parse_date(g("billPassedInRSDate")),
                "cmte": parse_date(g("referredToCommitteeDate")),
                "report": parse_date(g("reportPresentedDate")), "assent": parse_date(g("billAssentedDate")),
                "act_no": g("actNo"), "act_year": g("actYear"), "status": g("status"),
                "docs": [k for k, v in docs.items() if v]}

    def _sqlite(self):
        db = sqlite3.connect(":memory:")
        # The four *_m columns hold the year and month as "YYYY-MM" so a
        # question like "how many bills were introduced this month" can be
        # answered exactly. Without them only whole years could be counted.
        db.execute("CREATE TABLE bills(no TEXT, year TEXT, title TEXT, type TEXT, category TEXT,"
                   " ministry TEXT, intro TEXT, ls TEXT, rs TEXT, assent TEXT, status TEXT,"
                   " intro_m TEXT, ls_m TEXT, rs_m TEXT, assent_m TEXT)")
        iso = lambda d: d.isoformat() if d else None
        ym = lambda d: d.strftime("%Y-%m") if d else None
        db.executemany("INSERT INTO bills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       [(b["no"], b["year"], b["title"], b["type"], b["category"], b["ministry"],
                         iso(b["intro"]), iso(b["ls"]), iso(b["rs"]), iso(b["assent"]), b["status"],
                         ym(b["intro"]), ym(b["ls"]), ym(b["rs"]), ym(b["assent"]))
                        for b in self.bills])
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
        # "Bill No. 370 of 2019" / "bill 370/2019" / "370/2019". Needs the word "bill" BEFORE
        # the number, so a title like "Appropriation (No.3) Bill" is not misread as Bill No. 3.
        m = re.search(r"\bbill\s*(?:no\.?|number|num\.?)?\s*(\d{1,3})\b"
                      r"(?:\s*(?:of|/|,|\(|-)\s*(20[0-2]\d))?", q) or \
            re.search(r"\b(\d{1,3})\s*/\s*(20[0-2]\d)\b", q)
        if m:
            no, yr = m.group(1), (m.group(2) if m.lastindex and m.lastindex >= 2 else None)
            yr = yr or (years[-1] if years else None)
            if yr:
                b = self.by_key.get((no, yr))
                return {"kind": "one", "bills": [b], "others": []} if b else \
                    {"kind": "none", "bills": [], "others": [], "why": f"no Bill No. {no} of {yr}"}
            same = [b for b in self.bills if b["no"] == no]
            return {"kind": "many", "bills": same, "others": []} if same else \
                {"kind": "none", "bills": [], "others": []}
        qt = self._fix_typos(tokens(q, QUERY_NOISE))
        if qt:
            pool = [b for b in self.bills
                    if not years or b["year"] == years[-1] or years[-1] in b["title_years"]]
            qs = set(qt)
            cand = [b for b in pool if qs <= b["toks"]]
            if cand:
                cand.sort(key=lambda b: -len(qs) / len(qs | b["toks"]))
                top = len(qs) / len(qs | cand[0]["toks"])
                second = len(qs) / len(qs | cand[1]["toks"]) if len(cand) > 1 else 0
                if len(cand) == 1 or top - second >= 0.15:
                    return {"kind": "one", "bills": [cand[0]], "others": cand[1:4]}
                return {"kind": "many", "bills": cand, "others": []}
        if set(re.findall(r"[a-z]+", q)) & PRONOUNS and last_filters and all(last_filters):
            b = self.by_key.get((str(last_filters[1]), str(last_filters[0])))
            if b:
                return {"kind": "one", "bills": [b], "others": []}
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
        return f"**{b['title']}** (Bill No. {b['no']} of {b['year']})"

    def card(self, b):
        n = lambda d, empty="not recorded": fmt(d) or empty
        act = f" - Act No. {b['act_no']} of {b['act_year']}" if b["act_no"] else ""
        lines = [f"### {b['title']}", f"Bill No. {b['no']} of {b['year']} · {b['type'] or '?'} · "
                 f"{b['category'] or '?'} · Ministry: {b['ministry'] or 'not recorded'}",
                 "", "**Timeline**",
                 f"- Introduced ({b['house'] or 'house not recorded'}): {n(b['intro'])}"
                 + (f" by {b['by']}" if b["by"] else ""),
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
        has = lambda words: any(w in q for w in words)
        want = [k for k, w in FACT_WORDS.items() if has(w)]
        passed = bool(re.search(r"\bpass(ed|es)?\b", q))
        if "rajya" in q and passed:
            want.append("rs")
        if "lok sabha" in q and passed:
            want.append("ls")
        if passed and "rs" not in want and "ls" not in want:
            want += ["ls", "rs"]
        if re.search(r"\b(introduc\w*|moved|brought|tabled)\b", q) and "by" not in want:
            want.append("intro")
        out = []
        for k in dict.fromkeys(want):
            if k == "intro":
                out.append(f"Introduced in {b['house'] or 'the house'} on {fmt(b['intro'])}." if b["intro"]
                           else "No introduction date is recorded.")
            elif k == "ls":
                out.append(f"Passed by Lok Sabha on {fmt(b['ls'])}." if b["ls"] else
                           f"No Lok Sabha passing date is recorded (status: {b['status'] or 'unknown'}).")
            elif k == "rs":
                out.append(f"Passed by Rajya Sabha on {fmt(b['rs'])}." if b["rs"] else
                           "No Rajya Sabha passing date is recorded.")
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
                out.append(f"Introduced by {b['by']}." if b["by"] else "The mover is not recorded in the data.")
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
        return body + "\n\n" + SRC_LINE

    def many(self, res):
        bl = res["bills"]
        lines = [f"- {b['title']} - Bill No. {b['no']} of {b['year']} ({b['type']}, {b['status'] or '?'})"
                 for b in bl[:8]]
        more = f"\n...and {len(bl) - 8} more." if len(bl) > 8 else ""
        return (f"I found {len(bl)} bills that match, so I won't guess:\n" + "\n".join(lines) + more +
                "\n\nTell me the year or the bill number (e.g. \"Bill No. 370 of 2019\").")

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
        if re.search(r"assent|became law|become law|enacted|into law", q):
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
                        f"have a {dcol.upper()} date at all.")
            if rng and total:
                rows = self.db.execute(f"SELECT {col}, COUNT(*) FROM bills WHERE {w} "
                                       "GROUP BY 1 ORDER BY 1", args).fetchall()
                out += "\n\nBy year: " + ", ".join(f"{y}: {c}" for y, c in rows)
        else:
            rows = self.db.execute(f"SELECT title, no, year, status FROM bills WHERE {w} "
                                   "ORDER BY year, CAST(no AS INTEGER) LIMIT 12", args).fetchall()
            out = f"**{total} bills** match ({crit}; {basis})." + \
                  ("\n\n" + "\n".join(f"- {t} - Bill No. {n} of {y} ({s})" for t, n, y, s in rows)
                   if rows else "")
            if total > 12:
                out += f"\n...showing 12 of {total}. Add a year, type or ministry to narrow it."
        return out + "\n\n" + SRC_LINE

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
                # closest first: shortest title that still matched
                return sorted(group, key=lambda b: len(b["title"]))
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
        bills_db_selftest.run(db)


if __name__ == "__main__":
    main()
