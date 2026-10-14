#!/usr/bin/env python3
"""
STEP 3 - BUILD THE BRAIN (embeddings + search, NO LLM yet)
===========================================================
Turns clean chunks (Step 2) into a searchable brain:
ask a question -> get the exact official chunks that answer it.

  INPUT : ./clean_data/chunks.json
  OUTPUT: ./brain/
            embeddings.npy + cache_ids.json  vectors (built once, reused forever)
            cache_model.txt                  which model made those vectors
            fingerprint.json                 proves the stored brain matches this run
            chroma_db/                       searchable database on disk
            bm25.pkl                         word-match index
            retrieval_demo.txt               5 demo questions + answers (proof!)

Install (CPU-only, for normal student laptops - order matters!):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install sentence-transformers chromadb rank-bm25
  (GPU owners: just `pip install sentence-transformers chromadb rank-bm25`.)

Run   :  python step3_build_brain.py                 # full build + demo (~1-3 hrs once)
Test  :  python step3_build_brain.py --max-chunks 300  # small brain in minutes
Ask   :  python step3_build_brain.py --ask "What is the Data Protection Bill about?"
Chat  :  python step3_build_brain.py --interactive     # type questions, see chunks
Redo  :  python step3_build_brain.py --rebuild         # throw away brain, start over

Weak laptop? Add:  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (smaller, faster, slightly lower quality - same code, one flag.)
RAM note: bge-m3 needs ~4 GB free RAM. If yours runs out, the script
  says so plainly and tells you the one-flag fix (above). Nothing breaks.

VIVA ONE-LINERS (all true, all in this file):
  "Embeddings turn meaning into numbers, so similar ideas sit close together."
  "We filter by facts first - year, bill number - then search by meaning."
  "Hybrid search: vectors understand the question, keywords never miss names."
  "Step 3 has no LLM on purpose: we prove retrieval works BEFORE generating."
"""
import argparse
import json
import os
import pickle
import re
import sys
import time

os.environ["ANONYMIZED_TELEMETRY"] = "False"  # stop Chroma phoning home.

# ----------------------------------------------------------------------------
# CONFIG - every decision lives here.
# ----------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast on CPU (verified ~15 chunks/sec)
# To use the big multilingual model instead (bge-m3): add  --model BAAI/bge-m3
# to step3, step4_talk and app. It is MUCH slower on CPU (weeks, not hours).
SMALL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH = 32                  # embeddings at a time. Lower (16/8) if laptop struggles.
CHECKPOINT = 2000           # save progress every N chunks (a crash never loses hours)
EXCLUDE_SLOTS = {"errataFile"}  # typo-correction sheets = noise. Drop before embed.
RERANK_MODEL = "BAAI/bge-reranker-base"   # reads question + clause together
RERANK_POOL = 20          # candidates re-scored (more = slower, slightly better)
RERANK_TEXT = 400         # chars of a clause shown to the re-ranker

# A bill's title page is not an answer to anything. It carries the title,
# "AS PASSED BY THE HOUSES", the assent date, and nothing else. So a question
# like "what is <the title> about" matches it perfectly - the question IS the
# title - and it crowds out the Statement of Objects that actually explains the
# bill. Asked about the Juvenile Justice Bill, 2015, the top-scoring chunk was
# the assent line and the answer was procedural noise. These chunks are pushed
# down, never dropped: for a handful of bills they are the only text there is.
TITLE_PAGE_PENALTY = 0.35
# "BE it enacted" is deliberately NOT in this list. The operative half of every
# bill begins with the enacting formula, so matching it would demote clause 1 of
# every bill ever written. These two phrases appear only on the title page of a
# bill that has already been passed, which is exactly the page that carries no
# information about what the bill does.
TITLE_PAGE = re.compile(r"AS\s+PASSED\s+BY\s+THE\s+HOUSES|ASSENTED\s+TO\s+ON",
                        re.IGNORECASE)

# Why only 400 characters? Tested on the real clauses of Bill 82:
#   probe "what punishment ...", clause text capped at 1200 -> clause 8 won
#   probe "what punishment ...", clause text capped at  400 -> clause 4 won
#   probe "what subsistence ...", capped at 1200 -> clause 8 won
#   probe "what subsistence ...", capped at  400 -> clause 5 won
# A long clause carries a tail of legal boilerplate ("notwithstanding such
# repeal, anything done ... shall be deemed to have been done") that reads as
# substantive to the model. The opening of a clause is where its subject is.

VEC_W, KEY_W = 0.7, 0.3     # hybrid weights: meaning 70%, exact words 30%.
TOP_K = 14                  # candidates pulled from each search before merging
                            # (more candidates = a better chance the right clause
                            #  is found after duplicate texts are skipped).
TOP_N = 3                   # final answers shown.

WORD = re.compile(r"\w+", re.UNICODE)  # \w covers Hindi too (no special case!).
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
# Only an EXPLICIT bill number counts: "bill no. 370", "bill number 121",
# "bill 370", "bill 370 of 2019". Digits only, 1 to 4 long.
# The old pattern grabbed whatever words came after the word "bill"
# ("... Finance Bill 2024 about?" gave bill no "2024 about"). That filter
# matched no bill at all, so the search fell back to UNFILTERED and returned
# the wrong bill - the real cause of the Q08/Q12 style failures.
BILLNO = re.compile(r"\bbill\s*(?:no\.?|number|num\.?)?\s*(\d{1,4})\b"
                    r"(?:\s*of\s*(\d{4}))?", re.IGNORECASE)


# ----------------------------------------------------------------------------
# SMALL HELPERS - each does exactly one obvious thing.
# ----------------------------------------------------------------------------
def tokenize(s):
    """Lowercase words. Same function for index AND query (must match!)."""
    return WORD.findall((s or "").lower())

def strip_bill_name(question, bill_title=None):
    """
    Take the bill name and year out of a question, for the re-ranker.

    This matters more than it sounds. Once the bill is already known, its name
    adds nothing, and it actively hurts: every clause of Bill 82 is "about" the
    Muslim Women Bill, so the short title clause scores as high as the
    punishment clause.

    Tested directly on the real clauses:
      question kept whole   -> short title clause won (+0.063 vs +0.012)
      bill name removed     -> punishment clause won (+0.145 vs +0.027)
      "subsistence allowance" asked -> clause 5 won (+0.052 vs +0.001)
      "cognizable" asked          -> clause 7 won (+0.011 vs +0.0001)

    So the bill title has to be removed word by word, not just everything after
    the word "bill". "What punishment does the Muslim Women Bill 2019
    prescribe?" becomes "What punishment does the".
    """
    q = question or ""
    # the bill's own title, word by word, so "Muslim Women" goes too
    for word in re.findall(r"[a-z]{4,}", (bill_title or "").lower()):
        q = re.sub(r"\b" + re.escape(word) + r"\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(1[5-9]\d\d|20\d\d)\b", " ", q)
    q = re.sub(r"\bbill\b.*", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\bact\b.*", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(no|number)\b", " ", q, flags=re.IGNORECASE)
    # a bare number left over from "bill 82 of" is noise to the model
    q = re.sub(r"\b\d{1,4}\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ,.?")
    return q or (question or "")


def extract_filters(question):
    """Pulls exact facts from the question: year + bill number (if present)."""
    year, bill_no = None, None
    q = (question or "").strip().rstrip("?!.,")  # "What is bill 98?" -> "...bill 98"
    m = YEAR.search(q)
    if m and 1950 <= int(m.group(1)) <= 2030:
        year = m.group(1)
    # A four digit number in the year range is a YEAR, never a bill number, so
    # skip it and keep looking. "the Muslim Women Bill 2019 prescribe bill 82 of
    # 2019" used to stop at the first match and report bill number 2019, which
    # matched nothing and let the search drift to another bill of the same year.
    for m in BILLNO.finditer(q):
        n = m.group(1)
        if len(n) == 4 and n[:2] in ("19", "20"):
            continue
        bill_no = n
        break
    return year, bill_no

def load_model(model_name):
    """Loads the embedding model, with a friendly error if RAM is too small."""
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if the laptop has one
        return SentenceTransformer(model_name, device=device)
    except (RuntimeError, OSError) as e:
        if "mmap" in str(e) or "memory" in str(e).lower():
            sys.exit(
                "\nERROR: not enough free RAM for this model (it needs ~4 GB free).\n"
                "Close Chrome/other apps and retry, OR use the small model:\n"
                f"  python step3_build_brain.py --model {SMALL_MODEL}\n")
        raise

def dedupe_ids(chunks):
    """chunk_ids must be unique or Chroma refuses. Keep first, suffix rest."""
    seen, out = set(), []
    for c in chunks:
        cid, n = c["chunk_id"], 2
        while cid in seen:
            cid = f"{c['chunk_id']}-{n}"
            n += 1
        seen.add(cid)
        c["chunk_id"] = cid
        out.append(c)
    return out


# ----------------------------------------------------------------------------
# STAGE 1+2 - EMBED (with cache: never compute the same vector twice).
# ----------------------------------------------------------------------------
def get_embeddings(chunks, model_name, cache_dir):
    """Returns vectors aligned with chunks[]. Only embeds what's missing."""
    import numpy as np

    ids = [c["chunk_id"] for c in chunks]
    by_id = {c["chunk_id"]: c for c in chunks}  # dict lookup: instant at any size.
    npy_p = f"{cache_dir}/embeddings.npy"
    ids_p, mod_p = f"{cache_dir}/cache_ids.json", f"{cache_dir}/cache_model.txt"
    cache_ids, cache_vecs = [], None
    if os.path.exists(npy_p) and os.path.exists(ids_p):
        cache_ids = json.load(open(ids_p, encoding="utf-8"))
        cache_vecs = np.load(npy_p)
        print(f"      cache found: {len(cache_ids)} vectors on disk")
        old_model = open(mod_p, encoding="utf-8").read().strip() \
            if os.path.exists(mod_p) else None
        if old_model and old_model != model_name:  # vectors are model-specific!
            sys.exit(f"\nERROR: this brain was built with\n  {old_model}\n"
                     f"but you asked for\n  {model_name}\n"
                     "Mixing models silently corrupts search. Fix with ONE of:\n"
                     "  --rebuild  (rebuild everything with the new model), or\n"
                     "  --out ./brain2  (keep both brains), or\n"
                     f"  --model {old_model}  (reuse this brain)\n")

    missing = [i for i in ids if i not in set(cache_ids)]
    if missing:
        print(f"      embedding {len(missing)} new chunks with {model_name} ...")
        print("      (first run also downloads the model, one time only)")
        model = load_model(model_name)
        t0 = time.time()
        so_far = 0
        # Embed in slices and save after each slice. A crash or Ctrl+C can only
        # lose the current slice, and re-running resumes from the last save.
        for first in range(0, len(missing), CHECKPOINT):
            part = missing[first:first + CHECKPOINT]
            texts = [by_id[i]["text"] for i in part]
            new_vecs = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                                    show_progress_bar=True)
            cache_vecs = np.vstack([cache_vecs, new_vecs]) if cache_vecs is not None else new_vecs
            cache_ids = cache_ids + part
            np.save(npy_p, cache_vecs)
            json.dump(cache_ids, open(ids_p, "w", encoding="utf-8"))
            open(mod_p, "w", encoding="utf-8").write(model_name)
            so_far += len(part)
            print(f"      checkpoint saved: {so_far}/{len(missing)} chunks")
        dt = time.time() - t0
        print(f"      {len(missing)} chunks in {dt/60:.1f} min "
              f"({len(missing)/max(dt,1):.1f}/sec)")
    pos = {i: n for n, i in enumerate(cache_ids)}
    return [cache_vecs[pos[i]] for i in ids]


# ----------------------------------------------------------------------------
# STAGE 3 - STORE in Chroma (our vectors, explicitly - never Chroma's default!).
# ----------------------------------------------------------------------------
def build_chroma(chunks, vecs, db_dir, out_dir, model_name, rebuild):
    import chromadb
    # Fingerprint: reuse ONLY if same chunks AND same model. Count alone lies
    # (e.g. test run, then full run, then switching models - all look alike).
    fp = {"count": len(chunks), "first": chunks[0]["chunk_id"],
          "last": chunks[-1]["chunk_id"], "model": model_name}
    fp_p = f"{out_dir}/fingerprint.json"
    old_fp = json.load(open(fp_p, encoding="utf-8")) if os.path.exists(fp_p) else None
    client = chromadb.PersistentClient(path=db_dir)
    if rebuild:
        try:
            client.delete_collection("bills")
            print("      old brain deleted (--rebuild)")
        except Exception:
            pass  # fresh brain folder: there is nothing to delete yet. NOT an error.
        old_fp = None
    col = client.get_or_create_collection("bills", metadata={"hnsw:space": "cosine"})
    if old_fp == fp and col.count() == len(chunks):
        print(f"      Chroma already holds this exact brain ({col.count()} chunks) - reusing")
        return col
    if col.count() > 0:  # stale brain (test run / other model / other data): rebuild.
        print("      stored brain differs from this run - rebuilding Chroma ...")
        client.delete_collection("bills")
        col = client.get_or_create_collection("bills", metadata={"hnsw:space": "cosine"})
    print(f"      storing {len(chunks)} chunks in Chroma ...")
    for s in range(0, len(chunks), 1000):  # small batches: safe on any laptop.
        part = chunks[s:s + 1000]
        col.add(ids=[c["chunk_id"] for c in part],
                documents=[c["text"] for c in part],
                # "part" and "clause_no" are what let the talk layer read only
                # the law, or only the minister's explanation. They must be in
                # the database or a provision question can be answered from
                # background commentary.
                metadatas=[{k: c.get(k, "") for k in
                            ("bill_no", "bill_title", "year", "term", "ministry",
                             "bill_type", "status", "slot", "language",
                             "part", "clause_no", "label")
                            } for c in part],
                embeddings=[v.tolist() for v in vecs[s:s + 1000]])  # OUR vectors!
    json.dump(fp, open(fp_p, "w", encoding="utf-8"))
    print(f"      stored. DB lives in {db_dir}/ (persists forever)")
    return col


# ----------------------------------------------------------------------------
# STAGE 4 - KEYWORD index (classic BM25 word-match, one tiny file).
# ----------------------------------------------------------------------------
def build_bm25(chunks, path):
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([tokenize(c["text"]) for c in chunks])
    pickle.dump(bm, open(path, "wb"))
    print(f"      BM25 keyword index saved ({len(chunks)} docs)")
    return bm


# ----------------------------------------------------------------------------
# STAGE 5 - HYBRID SEARCH. The one function the whole project calls.
# ----------------------------------------------------------------------------
class Brain:
    """Loads everything once, answers questions forever."""

    def __init__(self, brain_dir, model_name, rerank=False):
        import chromadb
        # rerank=True loads a second, small model that reads the question and
        # one clause TOGETHER and scores whether the clause answers it. That is
        # a different question from "is this clause about the same subject",
        # which is all the vector search can answer. It costs about 1.3 seconds
        # on a CPU for 20 candidates, so it is off by default.
        self.rerank_on = bool(rerank)
        self._reranker = None
        if not os.path.exists(f"{brain_dir}/chroma_db"):
            sys.exit(f"\nERROR: no brain found in {brain_dir}/\n"
                     "Build it first:  python step3_build_brain.py --data out/chunks.json\n")
        kept = json.load(open(f"{brain_dir}/kept_chunks.json", encoding="utf-8"))
        self.chunks = {c["chunk_id"]: c for c in kept}
        self.order = [c["chunk_id"] for c in kept]
        # Every year this brain actually holds, so a question about a year
        # outside the corpus can be answered honestly instead of guessed at.
        self.years = {str(c.get("year")) for c in kept}
        self.col = chromadb.PersistentClient(path=f"{brain_dir}/chroma_db"
                                               ).get_collection("bills")
        self.bm = pickle.load(open(f"{brain_dir}/bm25.pkl", "rb"))
        if len(kept) != self.bm.corpus_size:  # hand-edited files? refuse loudly.
            sys.exit(f"\nERROR: brain files disagree in {brain_dir}/\n"
                     f"kept_chunks.json holds {len(kept)} chunks but bm25.pkl "
                     f"was built for {self.bm.corpus_size}.\n"
                     "Never hand-edit brain files. Fix: "
                     "python step3_build_brain.py --rebuild\n")
        print(f"      loading {model_name} (for question vectors) ...")
        self.model = load_model(model_name)

    @property
    def reranker(self):
        """Loaded on first use, so a brain without --rerank never downloads it."""
        if self._reranker is None and self.rerank_on:
            from sentence_transformers import CrossEncoder
            print("      loading %s (the re-ranker) ..." % RERANK_MODEL)
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        return self._reranker

    def chunks_for(self, bill_no, year):
        """
        EVERY chunk this brain holds for one bill, in the order the bill is
        written in.

        search() answers "which clause matches this question". This answers
        the other kind of question - "tell me about this bill" - where the
        right move is to read the whole thing rather than the three clauses
        that happened to score highest. The three best clauses of a
        hundred-clause bill are not a summary of it, they are three clauses.

        Returns [] when the bill has no readable text, which is also how a
        scanned bill is detected without a second lookup.
        """
        no, yr = str(bill_no), str(year)
        mine = [c for c in self.chunks.values()
                if str(c.get("bill_no")) == no and str(c.get("year")) == yr]

        def key(c):
            n = c.get("clause_no") or ""
            head = re.match(r"\d+", n)
            return (0 if c.get("part") == "operative" else 1,
                    int(head.group(0)) if head else 10 ** 6, n)

        return sorted(mine, key=key)

    def year_in_brain(self, year):
        """Is this year inside the corpus at all? A 1995 question in a
        2004-2026 brain must not be answered by silently dropping the filter."""
        if not year:
            return True
        return str(year) in self.years

    def search(self, question, top_n=TOP_N, part=None, rerank=None,
               filters=None):
        """Search the brain. `part` restricts to "operative" (the clauses, which
        is the law) or "narrative" (the Statement of Objects, which is the
        minister explaining it). Leave it None to search everything.

        A question about a punishment must not be answered from the minister's
        background note, and a question about why a bill was brought must not be
        answered from a clause, so the talk layer chooses which half to read.

        `filters` is (year, bill_no) supplied by a caller that has ALREADY
        worked out which bill is meant. Reading them back out of the question
        text is wrong whenever the two disagree, because a question often names
        a bill by the year printed in its TITLE while the bill itself is filed
        under the year it was introduced. "The Juvenile Justice (Care and
        Protection of Children) Bill, 2015" is Bill 99 of 2014, so the year in
        the sentence is 2015 and the bill is 2014.
        """
        # (a) facts first: year / bill-no filters (exact match, no guessing).
        year, bill_no = filters if filters else extract_filters(question)
        self.last_filters = (year, bill_no)  # remembered for follow-up questions.
        where = None
        conds = []
        if year and bill_no:
            conds = [{"year": year}, {"bill_no": bill_no}]
        elif year:
            conds = [{"year": year}]
        elif bill_no:
            conds = [{"bill_no": bill_no}]
        if part:
            conds.append({"part": part})
        if len(conds) == 1:
            where = conds[0]
        elif conds:
            where = {"$and": conds}
        # (b) meaning search (vectors) ...
        qvec = self.model.encode([question], normalize_embeddings=True)[0].tolist()
        res = self.col.query(query_embeddings=[qvec], n_results=TOP_K, where=where)
        filtered = bool(where)
        if not res["ids"][0] and where and self.year_in_brain(year):
            # Filter too strict? Retry unfiltered. But NOT when the year is
            # outside this brain's own years: a 1995 bill simply is not in a
            # 2004-2026 corpus, and searching unfiltered would answer a 1995
            # question with a 2024 bill. The part restriction always stays,
            # because a provision question must never fall back to narrative.
            retry_where = {"part": part} if part else None
            res = self.col.query(query_embeddings=[qvec], n_results=TOP_K,
                                 where=retry_where)
            filtered = False
        vec_scores = {i: max(0.0, 1.0 - d)  # cosine distance -> similarity.
                      for i, d in zip(res["ids"][0], res["distances"][0])}
        # (c) ... + word search (BM25) ...
        toks = tokenize(question)
        key_scores = {}
        if toks:
            scored = self.bm.get_scores(toks)
            best = sorted(range(len(scored)), key=lambda i: -scored[i])[:TOP_K]
            mx = scored[best[0]] if best and scored[best[0]] > 0 else 1.0
            for i in best:
                key_scores[self.order[i]] = float(scored[i] / mx)
        if filtered:  # keywords must vote INSIDE the filtered set too.
            # Otherwise a 2024 chunk could sneak into a 2019 answer, or a
            # narrative chunk could sneak into a provision answer. The vector
            # half was filtered by the database, but BM25 scores the whole
            # corpus, so its votes have to be checked here as well. Missing the
            # part check let the minister's background note score 0.3 from
            # keywords alone and outrank the real clause.
            for i in list(key_scores):
                c = self.chunks[i]
                if (year and c.get("year") != year):
                    del key_scores[i]
                elif (bill_no and c.get("bill_no") != bill_no):
                    del key_scores[i]
                elif (part and c.get("part") != part):
                    del key_scores[i]
        # (d) ... merged by weight. Both lists vote; best total wins.
        merged = {}
        for i in set(vec_scores) | set(key_scores):
            merged[i] = VEC_W * vec_scores.get(i, 0.0) + KEY_W * key_scores.get(i, 0.0)

        def rank_key(i):
            # The score shown to the user stays the honest merged score; only
            # the ORDER changes, so a title page can still win when it is the
            # only thing the bill has.
            s = merged[i]
            if TITLE_PAGE.search(self.chunks[i]["text"][:300]):
                s *= TITLE_PAGE_PENALTY
            return s

        ranked = sorted(merged, key=lambda i: -rank_key(i))

        # (d2) Re-rank. The merged score says how related a clause is to the
        # question. The re-ranker says whether it ANSWERS it. Those are
        # different, and only the second one is what we want: a punishment
        # question must be answered by the punishment clause, not by the short
        # title clause that merely belongs to the same bill.
        use_rerank = self.rerank_on if rerank is None else bool(rerank)
        if use_rerank and len(ranked) > 1:
            pool = ranked[:RERANK_POOL]
            rr = self.reranker
            if rr is not None:
                # Re-rank on the question with the bill name removed, because
                # the bill is already known and its name matches every clause
                # equally, which drowns out the part that discriminates.
                # Use the title of the top candidate, since the pool is
                # normally all from one bill once the filters have run.
                probe = strip_bill_name(
                    question, self.chunks[pool[0]].get("bill_title"))
                try:
                    scores = rr.predict(
                        [(probe, self.chunks[i]["text"][:RERANK_TEXT])
                         for i in pool])
                    order = sorted(range(len(pool)),
                                   key=lambda k: -float(scores[k]))
                    ranked = [pool[k] for k in order] + ranked[RERANK_POOL:]
                except Exception as e:
                    print("      (re-ranker failed, keeping vector order: %s)"
                          % type(e).__name__)

        # (e) Skip duplicate texts of the SAME bill. The same clause often sits
        # in several slots (gazette / passedLS / passedBoth), which wastes the
        # top-N on copies of one chunk. The first words are enough to spot a
        # copy, and the bill number keeps two different bills apart.
        chosen, seen_starts = [], set()
        for i in ranked:
            c = self.chunks[i]
            start = (c.get("bill_no"), c.get("year"),
                     " ".join(c["text"][:150].lower().split()))
            if start in seen_starts:
                continue
            seen_starts.add(start)
            chosen.append(i)
            if len(chosen) == top_n:
                break
        return [{"score": round(merged[i], 3), "chunk": self.chunks[i]} for i in chosen]


def show(hits):
    """Prints answers the way a human (and teacher) likes to read them."""
    for n, h in enumerate(hits, 1):
        c = h["chunk"]
        print(f"\n  [{n}] score={h['score']} | {c['bill_title'][:65]}")
        label = c.get("label") or ("clause " + c.get("clause_no", "")
                                   if c.get("clause_no") else c.get("part", ""))
        print(f"      Bill {c['bill_no']} ({c['year']}) - {c['ministry']} - "
              f"{c['status']} - {c['language']} - {c.get('part','')} - "
              f"{label[:50]}")
        print(f"      {c['text'][:320].replace(chr(10), ' ')}...")


# ----------------------------------------------------------------------------
# STAGE 6 - DEMO: questions built from YOUR data (works on test runs too).
# ----------------------------------------------------------------------------
def run_demo(brain, chunks, out_path):
    titles = list(dict.fromkeys(c["bill_title"] for c in chunks if c["bill_title"]))[:3]
    numeric = next((c for c in chunks
                    if c["bill_no"].strip().rstrip(".").isdigit()), chunks[0])
    questions = ([f"What is {t} about?" for t in titles]
                 + [f"Tell me about bill number {numeric['bill_no']} of {numeric['year']}"])
    if any(c["language"] in ("hindi", "mixed") for c in chunks):
        questions.append("यह विधेयक किस बारे में है?")  # Hindi proof (only if Hindi exists).
    lines = ["BILLS RETRIEVAL DEMO", "=" * 60]
    for q in questions:
        lines += ["", f"Q: {q}", "-" * 60]
        print(f"\nQ: {q}\n" + "-" * 60)
        hits = brain.search(q)
        show(hits)
        for h in hits:
            c = h["chunk"]
            lines += [f"[{h['score']}] {c['bill_no']}/{c['year']} {c['bill_title'][:70]}",
                      f"     {c['text'][:250].replace(chr(10), ' ')}..."]
    open(out_path, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nDemo saved to {out_path}  (screenshot this for your report!)")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 3: build the searchable brain")
    ap.add_argument("--data", default="./clean_data/chunks.json")
    ap.add_argument("--out", default="./brain")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--rerank", action="store_true",
                    help="re-score candidates with a cross-encoder (better "
                         "clause choice, about 1.3s more per question)")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--ask", default="", help="answer one question, no rebuild")
    ap.add_argument("--interactive", action="store_true")
    a = ap.parse_args()

    if a.ask or a.interactive:  # question mode: just load, never build.
        brain = Brain(a.out, a.model)
        if a.ask:
            show(brain.search(a.ask))
        else:
            print("Ask anything (type quit to exit). Example: What is the Finance Bill about?")
            while True:
                try:
                    q = input("\nQ: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in ("quit", "exit", ""):
                    break
                show(brain.search(q))
        return

    if not os.path.exists(a.data):
        sys.exit(f"\nERROR: {a.data} not found.\n"
                 "Run Step 2 first:  python step2_organize_clean.py\n"
                 "(it reads ./bills_data/ and writes ./clean_data/chunks.json)\n")
    print("\n[1/5] Loading chunks ...")
    chunks = [c for c in json.load(open(a.data, encoding="utf-8"))
              if c.get("slot") not in EXCLUDE_SLOTS]
    chunks = dedupe_ids(chunks)
    if a.max_chunks:
        chunks = chunks[:a.max_chunks]
    if not chunks:
        sys.exit("\nERROR: no chunks left to build from (all excluded or file empty).\n"
                 "Check quarantine.csv / failures in the Step 2 report.\n")
    print(f"      {len(chunks)} chunks (excluded slots: {sorted(EXCLUDE_SLOTS)})")
    os.makedirs(a.out, exist_ok=True)
    json.dump(chunks, open(f"{a.out}/kept_chunks.json", "w", encoding="utf-8"))

    print("[2/5] Embeddings ...")
    vecs = get_embeddings(chunks, a.model, a.out)

    print("[3/5] Storing in Chroma ...")
    build_chroma(chunks, vecs, f"{a.out}/chroma_db", a.out, a.model, a.rebuild)

    print("[4/5] Building keyword index ...")
    build_bm25(chunks, f"{a.out}/bm25.pkl")

    print("[5/5] Retrieval demo (no LLM yet - chunks only) ...")
    run_demo(Brain(a.out, a.model), chunks, f"{a.out}/retrieval_demo.txt")
    print("\nBrain ready. Next: Step 4 connects the LLM. Try now:")
    print('  python step3_build_brain.py --interactive')


if __name__ == "__main__":
    main()
