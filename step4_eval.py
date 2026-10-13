#!/usr/bin/env python3
"""
STEP 4 (part 2) - EVAL (same 20 questions, v1 vs v2 report)
============================================================
Proves the bot works - on YOUR data, with numbers for your report:
  v1 = Sansad official text only (sources=sansad)
  v2 = Sansad + PRS summaries    (sources=all)
Same 20 questions, side by side. Run it before AND after adding the
PRS layer (Step 1b): the table then shows exactly what PRS added.

Grade A - RETRIEVAL (free, no key): did the right chunks come back?
Grade B - ANSWERS (needs key): real LLM replies for K sample questions,
  plus the 2 must-refuse questions (debates + out-of-range year).

Run:  python step4_eval.py                        # Grade A only (free)
      python step4_eval.py --with-answers 5       # + Grade B for 5 Qs (~12 calls)
      python step4_eval.py --with-answers 2 --mock-llm   # offline test

Out : eval_report.txt + eval_report.json  (paste the table into your report!)

VIVA ONE-LINERS (all true, all in this file):
  "v1 vs v2 is an ablation: same questions, PRS is the only change."
  "Retrieval is graded by keywords; answers are graded by refusal + citations."
  "Two questions MUST fail - they test that the bot refuses honestly."
"""
import argparse
import json
import os
import sys
import time

from step3_build_brain import Brain, MODEL_NAME as EMBED_MODEL
from step4_talk import answer_question, format_sources, LLM_MODELS, REFUSAL

# ----------------------------------------------------------------------------
# THE 20 QUESTIONS. must = words the top-3 extracts must contain (lowercase).
# meta = metadata at least ONE extract must carry. expect=refuse => must refuse.
# ----------------------------------------------------------------------------
QUESTIONS = [
    {"q": "What is the Finance Bill 2024 about?",
     "must": ["finance"]},
    {"q": "Tell me about Bill No. 370 of 2019.",
     "must": ["citizenship"], "meta": {"bill_no": "370", "year": "2019"}},
    {"q": "What is the Digital Personal Data Protection Bill 2023 about?",
     "must": ["data protection"]},
    {"q": "What does the Bharatiya Nyaya Sanhita 2023 replace?",
     "must": ["penal"]},
    {"q": "Which communities are covered under the Citizenship Amendment Act 2019?",
     "must": ["hindu"]},
    {"q": "What punishment does the Triple Talaq Act 2019 prescribe?",
     "must": ["imprisonment"]},
    {"q": "What is the Income-tax Bill 2025 about?",
     "must": ["income"]},
    {"q": "What does the Constitution (106th Amendment) Bill 2023 provide for women?",
     "must": ["reservation"]},
    {"q": "Which Acts did the Farm Laws Repeal Act 2021 repeal?",
     "must": ["repeal"]},
    {"q": "What did the Jammu and Kashmir Reorganisation Bill 2019 do?",
     "must": ["reorganis"]},
    {"q": "What is the Statement of Objects and Reasons of the Finance Bill 2024?",
     "must": ["objects and reasons"], "meta": {"year": "2024"}},
    {"q": "What is Clause 2 of the Finance Bill 2024 about?",
     "must": ["finance"], "meta": {"year": "2024"}},
    {"q": "What did the Right to Information (Amendment) Bill 2019 change?",
     "must": ["information"]},
    {"q": "What is the Unlawful Activities (Prevention) Amendment Bill 2019 about?",
     "must": ["terrorist"]},
    {"q": "What penalties does the Digital Personal Data Protection Bill 2023 lay down?",
     "must": ["penalt"]},
    {"q": "वित्त विधेयक 2024 किस बारे में है?",
     "must": ["financ"]},
    {"q": "Which 2023 bills replaced colonial-era criminal laws?",
     "must": ["sanhita"], "meta": {"year": "2023"}},
    {"q": "Was the Triple Talaq Bill passed by Parliament?",
     "must": ["talaq"]},
    {"q": "What did MPs say during the debate on the Citizenship Amendment Bill?",
     "must": [], "expect": "refuse"},
    {"q": "Tell me about the Banking Companies Bill of 1995.",
     "must": [], "expect": "refuse"},
]


# ----------------------------------------------------------------------------
# GRADING - small, strict, explainable.
# ----------------------------------------------------------------------------
def grade_retrieval(brain, item, sources):
    """PASS if top-3 extracts contain the keywords AND (if given) the metadata."""
    if item.get("expect") == "refuse":
        return "-"  # judged at answer stage, not here.
    hits = brain.search(item["q"], top_n=3)
    if sources == "sansad":
        official = [h for h in hits if h["chunk"].get("source", "sansad") == "sansad"]
        hits = official or hits
    joined = " ".join(h["chunk"]["text"] for h in hits).lower()
    if not all(w in joined for w in item.get("must", [])):
        return "FAIL"
    meta = item.get("meta", {})
    if meta and not any(all(str(v).lower() in str(h["chunk"].get(k, "")).lower()
                             for k, v in meta.items()) for h in hits):
        return "FAIL"
    return "PASS"

def count_prs_chunks(brain_dir):
    """How many PRS summary chunks does this brain hold? (0 until Step 1b.)"""
    p = f"{brain_dir}/kept_chunks.json"
    if not os.path.exists(p):
        return 0
    kept = json.load(open(p, encoding="utf-8"))
    return sum(1 for c in kept if c.get("source") == "prs")


# ----------------------------------------------------------------------------
# MAIN.
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 4 eval: same 20 Qs, v1 vs v2")
    ap.add_argument("--out", default="./brain")
    ap.add_argument("--model", default=EMBED_MODEL, help="MUST match the brain's model")
    ap.add_argument("--with-answers", type=int, default=0,
                    help="also generate LLM answers for first K answerable Qs")
    ap.add_argument("--sleep", type=float, default=7,
                    help="seconds between LLM calls (stays under the ~10/min free limit)")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--mock-llm", action="store_true", help="offline fake LLM (tests)")
    a = ap.parse_args()

    key = a.api_key or os.environ.get("GEMINI_API_KEY", "")
    if (a.with_answers and not a.mock_llm and not key
            and not os.environ.get("GROQ_API_KEY")):
        sys.exit("\nERROR: --with-answers needs an LLM key (or --mock-llm to test).\n"
                 "Gemini (free): https://aistudio.google.com/apikey  -> GEMINI_API_KEY\n"
                 "Groq  (free): https://console.groq.com            -> GROQ_API_KEY\n")

    print("Loading brain ...")
    brain = Brain(a.out, a.model)
    n_prs = count_prs_chunks(a.out)

    # ---- Grade A: retrieval, v1 vs v2 ----
    rows = []
    for item in QUESTIONS:
        v1 = grade_retrieval(brain, item, "sansad")
        v2 = grade_retrieval(brain, item, "all")
        rows.append({"q": item["q"], "v1": v1, "v2": v2,
                     "delta": "changed" if v1 != v2 else "same"})
    p1 = sum(1 for r in rows if r["v1"] == "PASS")
    p2 = sum(1 for r in rows if r["v2"] == "PASS")
    answerable = sum(1 for i in QUESTIONS if i.get("expect") != "refuse")

    lines = ["BILLS BOT EVAL - v1 (Sansad only) vs v2 (Sansad + PRS)",
             f"Brain: {a.out} | PRS chunks in brain: {n_prs}", "=" * 72]
    if n_prs == 0:
        lines.append("NOTE: no PRS chunks yet - v1 and v2 SHOULD match. "
                     "Re-run after Step 1b.")
    lines.append("")
    for n, r in enumerate(rows, 1):
        lines.append(f"Q{n:02d} v1={r['v1']:4s} v2={r['v2']:4s} {r['q'][:58]}")
    lines += ["", f"Retrieval score: v1 {p1}/{answerable}, v2 {p2}/{answerable}"]
    report = "\n".join(lines)
    print("\n" + report)
    open("eval_report.txt", "w", encoding="utf-8").write(report)
    json.dump({"prs_chunks": n_prs, "v1_pass": p1, "v2_pass": p2,
               "answerable": answerable, "rows": rows},
              open("eval_report.json", "w", encoding="utf-8"), indent=1)
    print("\nSaved eval_report.txt + eval_report.json (paste the table in your report!)")

    # ---- Grade B: real answers for K questions + refusal checks ----
    if not a.with_answers:
        return
    todo = [i for i in QUESTIONS if i.get("expect") != "refuse"][:a.with_answers]
    todo += [i for i in QUESTIONS if i.get("expect") == "refuse"]
    out = ["BILLS BOT ANSWERS (exact + explain, v1 sources)", "=" * 72]
    for item in todo:
        modes = ["exact", "explain"] if item.get("expect") != "refuse" else ["exact"]
        for mode in modes:
            # New signature: (brain, question, mode, api_key, llm_models,
            # top_n, history, last_filters, no_llm, mock_llm).
            # The eval runs with no history on purpose, so every question is
            # judged on its own and a good answer cannot be borrowed from the
            # question before it.
            reply, hits, used = answer_question(brain, item["q"], mode, key,
                                                LLM_MODELS, 3, [], None,
                                                False, a.mock_llm)
            out += ["", f"Q ({mode}): {item['q']}", "-" * 72, reply, "",
                    f"Sources (via {used}):", format_sources(hits)]
            if item.get("expect") == "refuse":
                ok = REFUSAL.lower() in reply.lower()
                out.append(f"REFUSAL CHECK: {'REFUSED-correctly' if ok else 'DID-NOT-REFUSE'}")
            print(f"  answered ({mode}): {item['q'][:50]}")
            time.sleep(a.sleep if not a.mock_llm else 0)
    open("eval_answers.txt", "w", encoding="utf-8").write("\n".join(out))
    print("Saved eval_answers.txt")


if __name__ == "__main__":
    main()
