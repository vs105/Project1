
import argparse, csv, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

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


def safe_name(s, n=60):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", (s or "").strip())
    s = re.sub(r"\s+", " ", s)
    return s[:n].strip(" .") or "untitled"


def fetch_metadata(from_year, to_year):
    """Step 1: pull EVERY Lok Sabha bill (house sweep), keep in-scope years."""
    recs, page, total = [], 1, None
    while True:
        for attempt in range(3):  # hostel wifi drops: retry, don't die.
            try:
                r = requests.get(API, params={"house": "Lok Sabha", "page": page,
                                              "size": 500, "locale": "en"},
                                 headers=UA, timeout=30)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"  ... metadata page {page} failed, retrying ...")
                time.sleep(3)
        d = r.json()
        total = d["_metadata"]["totalElements"]
        for rec in d["records"]:
            try:
                y = int(rec.get("billYear") or 0)
            except (TypeError, ValueError):
                y = 0
            if from_year <= y <= to_year:
                recs.append(rec)
        print(f"  ... page {page}/{total // 500 + 1} ({len(recs)} in-scope so far)",
              end="\r")
        if page * 500 >= total:
            break
        page += 1
        time.sleep(0.2)
    print()
    return recs


def dedupe(recs):
    """Same bill listed twice (Passed + Assented) -> keep richest one."""
    best = {}
    for r in recs:
        # casefold: "CAA Bill" vs "caa bill" from the API = same bill, not two.
        key = (str(r.get("billNumber")).strip().casefold(),
               (r.get("billName") or "").strip().casefold(),
               str(r.get("billYear")))
        files = sum(1 for k in SLOTS if r.get(k))
        score = (files, STATUS_RANK.get(r.get("status"), 0))
        if key not in best or score > best[key][0]:
            best[key] = (score, r)
    return [v[1] for v in best.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2004)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--out", default="./bills_data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-slots", default="", help="e.g. errataFile")
    ap.add_argument("--fresh", action="store_true", help="re-download existing")
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip_slots.split(",") if s.strip()}
    os.makedirs(a.out, exist_ok=True)

    print(f"\n[1/3] Fetching Lok Sabha bills {a.from_year}-{a.to_year} ...")
    recs = fetch_metadata(a.from_year, a.to_year)
    print(f"      {len(recs)} raw records")
    recs = dedupe(recs)
    print(f"      {len(recs)} unique bills after dedupe")
    json.dump(recs, open(f"{a.out}/bills_metadata.json", "w", encoding="utf-8"), indent=1)

    cols = ["billNumber", "billName", "billType", "billCategory", "ministryName",
            "billYear", "billIntroducedInHouse", "billIntroducedBy",
            "billIntroducedDate", "billPassedInLSDate", "billPassedInRSDate",
            "referredToCommitteeDate", "reportPresentedDate", "actNo", "actYear",
            "billAssentedDate", "status"] + SLOTS
    with open(f"{a.out}/bills_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            row = {c: (str(r.get(c)).strip() if r.get(c) else "") for c in cols}
            w.writerow(row)
    print("      saved bills_metadata.json + .csv")

    # build download tasks
    tasks = []
    for i, r in enumerate(recs):
        for k in SLOTS:
            if not r.get(k) or k in skip:
                continue
            url = quote(r[k], safe=":/?&=%")  # spaces in filenames!
            ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
            ext = ext if ext in ("pdf", "doc", "docx", "htm", "html") else "pdf"
            fname = (f"{safe_name(str(r.get('billNumber')), 12)}_"
                     f"{r.get('billYear')}_{safe_name(r.get('billName'))}.{ext}")
            folder = f"{a.out}/pdfs/Y{a.from_year}-{a.to_year}/{SHORT[k]}"
            tasks.append({"url": url, "path": f"{folder}/{i:04d}_{fname}",
                          "bill": (r.get("billName") or "")[:60]})
    if a.max_files:
        tasks = tasks[:a.max_files]
    print(f"\n[2/3] Downloading {len(tasks)} original PDFs ... (Ctrl+C safe)")

    logf = open(f"{a.out}/download_log.csv", "a", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    if os.path.getsize(f"{a.out}/download_log.csv") == 0:
        logw.writerow(["url", "local_path", "status", "bytes"])
    lock, done = threading.Lock(), {"ok": 0, "skip": 0, "fail": 0}
    bar = tqdm(total=len(tasks), unit="file") if tqdm else None

    def looks_valid(path):
        """Magic-byte check: PDF / old-DOC / docx-ZIP / HTML. False if unreadable."""
        try:
            with open(path, "rb") as f:
                head = f.read(5)
        except OSError:
            return False
        return head.startswith((b"%PDF-", b"\xd0\xcf\x11\xe0", b"PK",
                                b"<htm", b"<HT"))

    def one(t):
        time.sleep(0.2)
        try:
            os.makedirs(os.path.dirname(t["path"]), exist_ok=True)
            if os.path.exists(t["path"]) and not a.fresh \
                    and os.path.getsize(t["path"]) > 1000 and looks_valid(t["path"]):
                st, n, which = "exists", os.path.getsize(t["path"]), "skip"
            else:
                tmp = t["path"] + ".part"  # crash-safe: half-files never pose as done.
                for attempt in range(3):
                    try:
                        # 120s + streaming: giant bills (600+ pages, 25MB) must
                        # never time out, even on slow hostel wifi.
                        r = requests.get(t["url"], headers=UA, timeout=120,
                                         stream=True)
                        r.raise_for_status()
                        total = 0
                        with open(tmp, "wb") as f:
                            for ch in r.iter_content(65536):
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

    print(f"\n[3/3] DONE. downloaded={done['ok']} already-had={done['skip']} "
          f"failed={done['fail']}")
    print(f"      Files : {a.out}/pdfs/")
    print(f"      Index : {a.out}/bills_metadata.csv  +  download_log.csv")
    if done["fail"]:
        print("      Re-run the same command to retry failures.")


if __name__ == "__main__":
    main()
