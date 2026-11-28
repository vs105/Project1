#!/usr/bin/env python3
"""
Lok Sabha + Rajya Sabha Bills - FULL MULTI-SOURCE DOWNLOADER
=============================================================
Downloads ORIGINAL bill PDFs + complete metadata, from TWO sources, and
merges them into ONE canonical dataset with a clear precedence rule.

WHY TWO SOURCES
---------------
No single source is complete. Measured against the Sansad API:

  * The Sansad API returns 3,813 Lok Sabha and 1,517 Rajya Sabha bills for
    2004-2026. Sweeping only house=Lok Sabha loses 1,517 bills - 29% of the
    real corpus - and 84% of those are Private Member bills.
  * The RTI Wiki CSV covers 2020-2026 only, but it holds 429 Rajya Sabha
    bills that the Lok Sabha sweep cannot see, and every one of its 1,409
    rows carries a PDF link.
  * On the 866 bills BOTH sources cover, they agree on every date, act
    number, ministry and type. The only disagreements are 10 status cases,
    all of them the RTI file being staler (it says "Passed" where Sansad
    says "Assented").

So cross-checking the two sources against each other finds almost nothing.
What merging buys is COVERAGE. That is why this script merges for coverage
and resolves conflicts by PRECEDENCE, never by voting.

PRECEDENCE RULE
---------------
    Sansad (official Parliament API) always wins.
    RTI Wiki only fills a field that Sansad left blank.

Every row records where it came from, so the chatbot can say "according to
official Parliament data" and be right:

    source_primary = sansad | rti      which source the bill was FIRST seen in
    filled_from    = comma list        which fields RTI filled in

THE UNIQUE KEY - read this before changing anything
---------------------------------------------------
    (house, billNumber, billYear)

NOT (house, billNumber, introductionYear). They are not the same thing:
744 of 3,813 Lok Sabha bills have billYear != introduction year, because a
bill introduced in February is numbered for the previous calendar year.
Measured: keyed on billYear, ZERO collisions in either house. Keyed on the
introduction year, 157 Rajya Sabha keys collide and 162 distinct bills get
silently merged into the wrong one.

RAJYA SABHA BILL NUMBERS ARE ROMAN NUMERALS
-------------------------------------------
Every one of the 1,517 Rajya Sabha bills is numbered "I", "II", ... "LXXXVIII"
- and 209 of them arrive from the API with trailing padding, like
"XXXVIII             ". The padding is stripped here, once, so that the CSV
and the JSON agree and nothing downstream has to guess.

WHAT YOU GET in ./bills_data/
    bills_metadata.json      all unique bills, both houses, both sources
    bills_metadata.csv       same, as a table, with provenance columns
    bills_metadata_merge.csv what the merge did: added, filled, skipped
    pdfs/Y2004-2026/<slot>/  PDFs organised by year-range + stage
    download_log.csv         every file: source URL -> local path -> status

Needs ~3 GB free disk. Resume-safe: press Ctrl+C anytime, re-run to continue.

Install once :  pip install requests tqdm
Run :           python download_bills.py
Test first :    python download_bills.py --max-files 10 --out ./test_data
Skip the merge:  python download_bills.py --no-rti
Sansad only, one house:  python download_bills.py --houses "Lok Sabha"
"""
import argparse, csv, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

# The Roman -> Arabic converter lives in bills_db.py, so the downloader, the
# facts layer and the brain all agree on what "Bill XXXVIII" and "Bill 38" mean.
# One copy, not three that can drift apart.
try:
    from bills_db import normalize_bill_number, norm_key as _nk
except ImportError as _e:
    sys.exit("ERROR: bills_db.py is missing from this folder (%s).\n"
             "Copy bills_db.py next to download_bills.py and run again." % _e)

try:
    import requests
except ImportError:
    sys.exit("ERROR: run  pip install requests tqdm  first")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # fallback to plain prints

API = "https://sansad.in/api_rs/legislation/getBills"
UA = {"User-Agent": "Mozilla/5.0 (student-project; academic-use)"}
RTI_URL = ("https://righttoinformation.wiki/static/data/csv/"
           "parliament-bills-2020-2026.csv")

SLOTS = ["billIntroducedFile", "billPassedInLSFile", "billPassedInRSFile",
         "billPassedInBothHousesFile", "errataFile", "reportFile",
         "billGazettedFile", "billSynopsisFile"]
SHORT = {"billIntroducedFile": "intro", "billPassedInLSFile": "passedLS",
         "billPassedInRSFile": "passedRS", "billPassedInBothHousesFile": "passedBoth",
         "errataFile": "errata", "reportFile": "committeeReport",
         "billGazettedFile": "gazette", "billSynopsisFile": "synopsis"}
STATUS_RANK = {"Assented": 6, "Passed": 5, "Pending": 4, "Part-discussed": 3,
               "Withdrawn": 2, "Negatived": 2, "Lapsed": 1, "Removed": 1,
               "Repealed": 1, None: 0}

# Every column the rest of the pipeline reads, in the order the CSV gets.
# house / source_primary / filled_from are the provenance columns; billYear is
# part of the unique key and must never be derived from the introduction date.
# bill_number_norm is the integer form of the bill number: "XXXVIII" -> 38.
# The raw billNumber is kept as well, because the Roman form is the OFFICIAL one
# and must be shown to the user. See bills_db.normalize_bill_number for why the
# house has to travel with it.
COLS = ["house", "billNumber", "bill_number_norm", "billYear", "billName", "billType",
        "billCategory", "ministryName", "billIntroducedInHouse",
        "billIntroducedBy", "billIntroducedDate", "billPassedInLSDate",
        "billPassedInRSDate", "referredToCommitteeDate",
        "reportPresentedDate", "actNo", "actYear", "billAssentedDate",
        "status"] + SLOTS + ["source_primary", "filled_from"]

# RTI Wiki column -> Sansad column. Only the fields RTI actually carries.
RTI_MAP = {"bill_number": "billNumber", "bill_name": "billName",
           "year": "billYear", "type": "billType",
           "category": "billCategory", "ministry": "ministryName",
           "introduced_in": "billIntroducedInHouse",
           "introduced_by": "billIntroducedBy",
           "introduced_date": "billIntroducedDate",
           "passed_ls_date": "billPassedInLSDate",
           "passed_rs_date": "billPassedInRSDate",
           "assented_date": "billAssentedDate", "status": "status",
           "committee_referred": "referredToCommitteeDate",
           "committee_report_date": "reportPresentedDate",
           "act_no": "actNo", "act_year": "actYear",
           "introduced_file": "billIntroducedFile",
           "report_file": "reportFile",
           "gazette_file": "billGazettedFile"}


def safe_name(s, n=60):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", (s or "").strip())
    s = re.sub(r"\s+", " ", s)
    return s[:n].strip(" .") or "untitled"


def norm_title(s):
    """Lowercase, drop punctuation, collapse spaces - for comparing titles."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def key_of(rec):
    """
    THE unique key: (house, bill_number_norm, billYear).

    bill_number_norm is the INTEGER form of the number, so Rajya Sabha
    "XXXVIII" and Lok Sabha "38" both become 38 -- which is exactly why the
    house is in the key. Measured on the whole corpus: (norm, year) alone has
    1,289 collisions, because Lok Sabha Bill 10 of 2004 and Rajya Sabha Bill X
    of 2004 are both number 10. (house, norm, year) has 0. Dropping the house
    here would turn 5,169 unambiguous bills into 1,289 ambiguous ones.

    billYear is used as-is - it is the number the bill was FILED under, NOT the
    year it was introduced. Keying on the introduction year is a bug: the two
    "Bill XXXVIII of 2025" are Bill XXXVIII of 2024 and of 2025.
    """
    return _nk(rec.get("house"), rec.get("bill_number_norm"),
               rec.get("billNumber"), rec.get("billYear"))


# ------------------------------------------------------------ sansad fetch --

def fetch_metadata(house, from_year, to_year):
    """Pull EVERY bill of one house (house sweep), keep in-scope billYear."""
    recs, page, total = [], 1, None
    while True:
        for attempt in range(3):  # hostel wifi drops: retry, don't die.
            try:
                r = requests.get(API, params={"house": house, "page": page,
                                              "size": 500, "locale": "en"},
                                 headers=UA, timeout=30)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"  ... {house} page {page} failed, retrying ...")
                time.sleep(3)
        d = r.json()
        total = d["_metadata"]["totalElements"]
        for rec in d["records"]:
            try:
                y = int(rec.get("billYear") or 0)
            except (TypeError, ValueError):
                y = 0
            if not (from_year <= y <= to_year):
                continue
            # Normalise ONCE, here, so nothing downstream has to guess. The raw
            # string stays (it is the official form and gets displayed) and the
            # integer is what every key and every lookup uses.
            rec["billNumber"], rec["bill_number_norm"] = normalize_bill_number(
                rec.get("billNumber"))
            rec["house"] = house
            rec["source_primary"] = "sansad"
            recs.append(rec)
        print(f"  ... {house} page {page}/{total // 500 + 1} "
              f"({len(recs)} in-scope so far)", end="\r")
        if page * 500 >= total:
            break
        page += 1
        time.sleep(0.2)
    print()
    return recs


def fetch_rti(from_year, to_year):
    """
    Read the RTI Wiki CSV and reshape it into the Sansad field names.

    This is a COVERAGE source, not an authority. It is fetched second and it
    can only ever fill a blank or add a bill the official API does not have.
    """
    for attempt in range(3):
        try:
            r = requests.get(RTI_URL, headers=UA, timeout=90)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                # A silent failure here would look exactly like "RTI added
                # nothing". Print the reason so the run log shows the difference.
                print("      RTI Wiki unreachable (%s) - continuing with Sansad only" % e)
                return []
            time.sleep(3)
    text = r.content.decode("utf-8", "replace")
    out = []
    for row in csv.DictReader(text.splitlines()):
        try:
            y = int(row.get("year") or 0)
        except (TypeError, ValueError):
            continue
        if not (from_year <= y <= to_year):
            continue
        house = (row.get("introduced_in") or "").strip()
        if house.casefold() not in ("lok sabha", "rajya sabha"):
            print("      RTI row skipped: House missing/unknown for %s (%s)"
                  % (row.get("bill_name") or "unnamed bill",
                     row.get("bill_number") or "?"))
            continue
        rec = {"house": "Lok Sabha" if house.casefold() == "lok sabha"
               else "Rajya Sabha", "source_primary": "rti"}
        for src, dst in RTI_MAP.items():
            v = (row.get(src) or "").strip()
            if v:
                rec[dst] = v
        if not rec.get("billNumber") or not rec.get("billName"):
            continue
        rec["billNumber"], rec["bill_number_norm"] = normalize_bill_number(
            rec["billNumber"])
        out.append(rec)
    return out


# ------------------------------------------------------------------- merge --

def dedupe(recs):
    """Deduplicate only identical bill identities; never hide title conflicts."""
    best = {}
    for source in recs:
        r = dict(source)
        raw, number_norm = normalize_bill_number(r.get("billNumber"))
        number_key = str(number_norm) if number_norm is not None else raw.casefold()
        house = str(r.get("house") or "").strip().casefold()
        year = str(r.get("billYear") or "").strip()
        if not house or not number_key or not year:
            raise ValueError("Bill metadata lacks House/number/year: %r" % r)
        key = (house, number_key, year)
        current = best.get(key)
        if current is None:
            best[key] = r
            continue
        old_title = norm_title(current.get("billName"))
        new_title = norm_title(r.get("billName"))
        if old_title and new_title and old_title != new_title:
            raise ValueError(
                "Conflicting titles for the same bill identity %r: %r vs %r" %
                (key, current.get("billName"), r.get("billName")))
        # Retain all non-conflicting fields from either copy. For conflicting
        # populated values, keep the record with more document links / stronger
        # status; this is still the same identity and title.
        old_score = (sum(bool(current.get(k)) for k in SLOTS),
                     STATUS_RANK.get(current.get("status"), 0))
        new_score = (sum(bool(r.get(k)) for k in SLOTS),
                     STATUS_RANK.get(r.get("status"), 0))
        winner, other = (r, current) if new_score > old_score else (current, r)
        merged = dict(winner)
        for col, value in other.items():
            if value and not merged.get(col):
                merged[col] = value
        best[key] = merged
    return list(best.values())


def merge(sansad, rti):
    """
    Merge RTI into Sansad by precedence. Returns (merged, report rows).

    Three things can happen to an RTI row:
      ADDED   - no Sansad bill shares (house, number, billYear) and no Sansad
                bill of that year has a near-identical title.
      FILLED  - it matches a Sansad bill, and it supplies fields Sansad left
                blank. Sansad's own values are never overwritten.
      SKIPPED - it matches a Sansad bill and adds nothing new.

    The title guard is what stops a merge double-counting: 15 of the 411
    RTI-only bills carry a title that already exists in Sansad under a
    different number.
    """
    by_key = {key_of(r): r for r in sansad}
    by_title = {}
    for r in sansad:
        by_title.setdefault((str(r.get("house") or "").strip().casefold(),
                             str(r.get("billYear") or "").strip(),
                             norm_title(r.get("billName"))), r)

    # NOT [dict(r) for r in sansad]. by_key below must point at the SAME
    # dict objects that end up in the output, otherwise the FILLED writes
    # land on copies that are thrown away: the report would say "filled
    # billPassedInLSDate" while the CSV still shows the column empty.
    merged = list(sansad)
    report = []
    for r in rti:
        k = key_of(r)
        t = (str(r.get("house") or "").strip().casefold(),
             str(r.get("billYear") or "").strip(),
             norm_title(r.get("billName")))
        hit = by_key.get(k)
        same_title = by_title.get(t)
        if hit is None and same_title is not None:
            report.append(["SKIPPED", r.get("house"), r.get("billNumber"),
                           r.get("billYear"), "duplicate title of Bill %s of %s"
                           % (same_title.get("billNumber"), same_title.get("billYear")),
                           (r.get("billName") or "")[:60]])
            continue
        if hit is None:
            row = dict(r)
            row["filled_from"] = "all (new bill)"
            merged.append(row)
            by_key[k] = row
            by_title[t] = row
            report.append(["ADDED", r.get("house"), r.get("billNumber"),
                           r.get("billYear"), "not in Sansad",
                           (r.get("billName") or "")[:60]])
            continue
        filled = []
        for col, val in r.items():
            if col in ("house", "source_primary", "filled_from"):
                continue
            if val and not hit.get(col):
                hit[col] = val
                filled.append(col)
        if filled:
            hit["filled_from"] = ",".join(sorted(filled))
            report.append(["FILLED", hit.get("house"), hit.get("billNumber"),
                           hit.get("billYear"), ",".join(sorted(filled)),
                           (hit.get("billName") or "")[:60]])
        else:
            report.append(["SKIPPED", hit.get("house"), hit.get("billNumber"),
                           hit.get("billYear"), "Sansad already complete",
                           (hit.get("billName") or "")[:60]])
    return merged, report


def repair_legacy(r):
    """
    Fill in the identity fields on a row saved by an OLDER downloader.

    An index written before the house column existed carries house="" and
    bill_number_norm="". key_of() then produces ('', '10', '2004') where a
    fresh row produces ('lok sabha', '10', '2004') - two different keys for
    the SAME bill, so the merge keeps the old row as well and the index ends
    up holding every one of those bills twice. A duplicated bill answers
    "I found 2 bills that match, so I won't guess", which silently loses it.

    The house is recovered from the FORM of the number, which is measured on
    the whole corpus and has zero exceptions either way: all 3,678 Lok Sabha
    numbers are Arabic, all 1,491 Rajya Sabha numbers are Roman. This is only
    a fallback for rows that are missing the field - a freshly fetched row
    always carries its house, and the merge rule already lets the fresh row
    win on any conflict, so a wrong guess here can only ever be overridden.
    """
    if (r.get("house") or "").strip() and (r.get("bill_number_norm") or "").strip():
        return r
    raw = (r.get("billNumber") or "").strip()
    if not raw:
        return r
    if not (r.get("bill_number_norm") or "").strip():
        _raw, n = normalize_bill_number(raw)
        if n is not None:
            r["bill_number_norm"] = str(n)
    if not (r.get("house") or "").strip():
        # The house was ALWAYS in the API row, under its own column name. Only
        # the derived "house" column is new, so read the original first and
        # fall back to the numeral form only when that is blank too.
        orig = (r.get("billIntroducedInHouse") or "").strip()
        if orig:
            r["house"] = orig
        else:
            r["house"] = "Lok Sabha" if raw.isdigit() else "Rajya Sabha"
    return r


def merge_with_disk(recs, out_dir):
    """
    Union this run's rows with the rows already saved in out_dir.

    Why this exists: the downloader is resume-safe for FILES (an existing
    PDF is never re-fetched), so it is natural to run it in pieces -- one
    house, or one year range, at a time. Writing the index in "w" mode
    throws away every row from the earlier pieces, and the PDFs left on
    disk then have no metadata. Step 2 silently drops them.

    Rule: freshly fetched rows win on a conflict, because they came from
    the API just now. Rows this run did not touch are preserved untouched.
    """
    p = os.path.join(out_dir, "bills_metadata.csv")
    if not os.path.exists(p):
        return recs
    try:
        with open(p, encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    except Exception as e:
        print("      ! could not read the existing index (%s) - rewriting" % e)
        return recs
    if not old:
        return recs
    by_key = {}
    for row in recs:
        k = key_of(row)
        if k in by_key and norm_title(by_key[k].get("billName")) != norm_title(row.get("billName")):
            raise ValueError("Conflicting fresh metadata rows for identity %r" % (k,))
        by_key[k] = dict(row)
    added = repaired = collapsed = 0
    for row in old:
        was_legacy = not (row.get("house") or "").strip()
        r = repair_legacy(dict(row))
        if was_legacy:
            repaired += 1
        k = key_of(r)
        if k not in by_key:
            by_key[k] = dict(r)
            added += 1
        else:
            old_title = norm_title(by_key[k].get("billName"))
            disk_title = norm_title(r.get("billName"))
            if old_title and disk_title and old_title != disk_title:
                raise ValueError("Existing metadata conflicts with fresh row for identity %r: %r vs %r" %
                                 (k, by_key[k].get("billName"), r.get("billName")))
            # Fresh data wins on populated conflicts. The older row may still
            # carry useful non-conflicting fields omitted by the current API.
            for col, value in r.items():
                if value and not by_key[k].get(col):
                    by_key[k][col] = value
            if was_legacy:
                collapsed += 1
    out = list(by_key.values())
    print("      index merged with disk: %d from this run, %d kept from "
          "before, %d total" % (len(recs), added, len(out)))
    if repaired:
        print("      repaired %d row(s) saved by an older downloader "
              "(no house column): %d collapsed into this run, %d kept"
              % (repaired, collapsed, repaired - collapsed))
    return out


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2004)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--out", default="./bills_data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-slots", default="", help="e.g. errataFile")
    ap.add_argument("--fresh", action="store_true", help="re-download existing")
    ap.add_argument("--houses", default="Lok Sabha,Rajya Sabha",
                    help='comma separated, e.g. "Lok Sabha"')
    ap.add_argument("--no-rti", action="store_true",
                    help="skip the RTI Wiki merge")
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip_slots.split(",") if s.strip()}
    houses = [h.strip() for h in a.houses.split(",") if h.strip()]
    os.makedirs(a.out, exist_ok=True)

    print(f"\n[1/4] Fetching bills {a.from_year}-{a.to_year} from "
          f"{', '.join(houses)} ...")
    recs = []
    for house in houses:
        got = fetch_metadata(house, a.from_year, a.to_year)
        print(f"      {house}: {len(got)} unique bills")
        recs.extend(got)
    recs = dedupe(recs)
    print(f"      {len(recs)} bills after dedupe")

    report = []
    if not a.no_rti:
        print("\n[2/4] Fetching RTI Wiki (2020-2026 coverage source) ...")
        rti = fetch_rti(a.from_year, a.to_year)
        print(f"      {len(rti)} RTI rows")
        recs, report = merge(recs, rti)
        n_add = sum(1 for r in report if r[0] == "ADDED")
        n_fill = sum(1 for r in report if r[0] == "FILLED")
        n_skip = sum(1 for r in report if r[0] == "SKIPPED")
        print(f"      merged: {len(recs)} bills  "
              f"(added {n_add}, filled {n_fill}, skipped {n_skip})")
        # Cumulative, for the same reason the index is: a per-house or
        # per-year run must not erase what the earlier runs recorded.
        rp = f"{a.out}/bills_metadata_merge.csv"
        prev = []
        if os.path.exists(rp):
            try:
                with open(rp, encoding="utf-8") as f:
                    prev = list(csv.reader(f))[1:]
            except Exception:
                prev = []
        seen = {(r[1], r[2], r[3]) for r in report if len(r) > 3}
        for r in prev:
            if len(r) > 3 and (r[1], r[2], r[3]) not in seen:
                report.append(r)
        with open(rp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["action", "house", "billNumber", "billYear",
                        "what", "title"])
            w.writerows(report)
    else:
        print("\n[2/4] --no-rti given, Sansad only")

    # Sort so the file is stable and diffable between runs.
    recs.sort(key=lambda r: (str(r.get("house") or ""),
                             int(r.get("billYear") or 0),
                             str(r.get("billNumber") or "")))
    # MERGE with what is already on disk. The PDFs are resume-safe -- an
    # existing file is never re-downloaded -- so the index must be too.
    # Without this, sweeping one house at a time
    # (   --houses "Rajya Sabha"  then  --houses "Lok Sabha"  )
    # wipes the first run's rows and leaves Rajya Sabha PDFs on disk with no
    # metadata, which Step 2 then reports as
    # "14 of 28 files did not match a metadata row".
    # Freshly fetched rows win on conflict; rows this run never saw are kept.
    recs = merge_with_disk(recs, a.out)
    json.dump(recs, open(f"{a.out}/bills_metadata.json", "w",
                         encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(f"{a.out}/bills_metadata.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            # Every value is stripped: Rajya Sabha bill numbers arrive from
            # the API padded with spaces, and an unstripped key matches nothing.
            w.writerow({c: (str(r.get(c)).strip() if r.get(c) else "")
                        for c in COLS})
    print(f"      saved bills_metadata.json + .csv  ({len(recs)} bills)")

    # ---- download the PDFs ----
    tasks = []
    # Build one task list PER HOUSE, then interleave them round-robin.
    # recs is sorted by house, so a plain loop would put all 3,678 Lok Sabha
    # bills first and all 1,491 Rajya Sabha bills last. With --max-files for
    # a quick test that means the run looks 100% Lok Sabha and you cannot
    # tell whether the Rajya Sabha path works at all. Round-robin makes any
    # capped run representative of the whole corpus.
    per_house = {}
    for r in recs:
        per_house.setdefault(str(r.get("house") or ""), []).append(r)
    order = []
    if per_house:
        lanes = list(per_house.values())
        while any(lanes):
            for lane in lanes:
                if lane:
                    order.append(lane.pop(0))
    i = 0
    for r in order:
        for k in SLOTS:
            if not r.get(k) or k in skip:
                continue
            url = quote(r[k], safe=":/?&=%")  # spaces in filenames!
            ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
            if ext not in ("pdf", "doc", "docx"):
                print("      skipping unsupported document format: %s (%s)"
                      % (r.get("billName") or "bill", ext))
                continue
            # The bill number can be Roman ("XXXVIII"), so it is kept as a
            # safe string, not forced to digits.
            num = safe_name(str(r.get("billNumber")), 16) or "NA"
            fname = f"{num}_{r.get('billYear')}_{safe_name(r.get('billName'))}.{ext}"
            folder = f"{a.out}/pdfs/Y{a.from_year}-{a.to_year}/{SHORT[k]}"
            tasks.append({"url": url, "path": f"{folder}/{i:05d}_{fname}",
                          "bill": (r.get("billName") or "")[:60]})
        i += 1
    if a.max_files:
        tasks = tasks[:a.max_files]
    print(f"\n[3/4] Downloading {len(tasks)} original PDFs ... (Ctrl+C safe)")

    logf = open(f"{a.out}/download_log.csv", "a", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    if os.path.getsize(f"{a.out}/download_log.csv") == 0:
        logw.writerow(["url", "local_path", "status", "bytes"])
    lock, done = threading.Lock(), {"ok": 0, "skip": 0, "fail": 0}
    bar = tqdm(total=len(tasks), unit="file") if tqdm else None

    def looks_valid(path):
        """Magic-byte check: PDF / old-DOC / docx-ZIP / HTML."""
        try:
            with open(path, "rb") as f:
                head = f.read(5)
        except OSError:
            return False
        return head.startswith((b"%PDF-", b"\xd0\xcf\x11\xe0", b"PK"))

    def one(t):
        time.sleep(0.2)
        try:
            os.makedirs(os.path.dirname(t["path"]), exist_ok=True)
            if os.path.exists(t["path"]) and not a.fresh \
                    and os.path.getsize(t["path"]) > 1000 and looks_valid(t["path"]):
                st, n, which = "exists", os.path.getsize(t["path"]), "skip"
            else:
                tmp = t["path"] + ".part"  # crash-safe.
                for attempt in range(3):
                    try:
                        # 120s + streaming: giant bills (600+ pages, 25MB) must
                        # never time out, even on slow hostel wifi.
                        total = 0
                        with requests.get(t["url"], headers=UA, timeout=120,
                                          stream=True) as r:
                            r.raise_for_status()
                            with open(tmp, "wb") as f:
                                for ch in r.iter_content(65536):
                                    if ch:
                                        f.write(ch)
                                        total += len(ch)
                        if total < 1000 or not looks_valid(tmp):
                            raise ValueError("not a valid document")
                        os.replace(tmp, t["path"])  # atomic: done means DONE.
                        st, n, which = "ok", total, "ok"
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 * (attempt + 1))
            with lock:
                done[which] += 1
                logw.writerow([t["url"], t["path"], st, n])
                logf.flush()
        except Exception as e:
            with lock:
                logw.writerow([t["url"], t["path"], f"FAIL:{str(e)[:80]}", 0])
                logf.flush()
                done["fail"] += 1
        finally:
            with lock:
                if bar:
                    bar.update(1)
                elif sum(done.values()) % 50 == 0:
                    print(f"      ... {sum(done.values())}/{len(tasks)}")

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(one, tasks))
    except KeyboardInterrupt:
        print("\n      Paused by you. Re-run anytime to resume.")
    finally:
        if bar:
            bar.close()
        logf.close()

    print(f"\n[4/4] DONE. downloaded={done['ok']} already-had={done['skip']} "
          f"failed={done['fail']}")
    print(f"      Files : {a.out}/pdfs/")
    print(f"      Index : {a.out}/bills_metadata.csv  +  download_log.csv")
    if report:
        print(f"      Merge : {a.out}/bills_metadata_merge.csv")
    if done["fail"]:
        print("      Re-run the same command to retry failures.")


if __name__ == "__main__":
    main()
