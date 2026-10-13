"""Self-test + auto-generated evaluation for bills_db.py.
Ground truth comes straight from the CSV rows, so hundreds of questions are graded
with NO human labelling and NO API key. Verdicts: CORRECT / CLARIFIED / WRONG / MISSED."""
import random
from bills_db import fmt

FIXED = [  # (question, text that MUST appear)
    ("When was Bill No. 370 of 2019 passed?", "10 December 2019"),
    ("when was bill no 370 of 2019 assented", "12 December 2019"),
    ("CAA passed when", "10 December 2019"),
    ("citizenship amandment bill 2019 passed date", "10 December 2019"),
    ("Tell me everything about Bill 370 of 2019", "Act No. 47 of 2019"),
    ("which ministry introduced the DPDP bill", "ELECTRONICS AND INFORMATION TECHNOLOGY"),
    ("when was the Appropriation (No.3) Bill, 2005 passed", "Appropriation (No.3) Bill, 2005"),
    ("when was the citizenship amendment bill passed", "won't guess"),
    ("when was BNS passed", "won't guess"),
]

def run(db, n=300, seed=7):
    bad = 0
    for q, must in FIXED:
        r, _ = db.answer(q)
        ok = bool(r) and must.lower() in r.lower()
        bad += (not ok)
        print(("PASS " if ok else "FAIL ") + q)
    r, _ = db.answer("what does the Finance Bill 2024 say about tax")
    ok = r is None
    bad += (not ok)
    print(("PASS " if ok else "FAIL ") + "content question falls through to RAG")
    rnd = random.Random(seed)
    pool = [b for b in db.bills if b["ls"]]
    sample = rnd.sample(pool, min(n, len(pool)))
    for label, make in (("by bill number", lambda b: f"When was Bill No. {b['no']} of {b['year']} passed by Lok Sabha?"),
                        ("by title", lambda b: f"When was {b['title']} passed by Lok Sabha?")):
        tally = {"CORRECT": 0, "CLARIFIED": 0, "WRONG": 0, "MISSED": 0}
        wrong = []
        for b in sample:
            r, _ = db.answer(make(b))
            if r is None:
                tally["MISSED"] += 1
            elif fmt(b["ls"]) in r and b["title"] in r:
                tally["CORRECT"] += 1
            elif "won't guess" in r:
                tally["CLARIFIED"] += 1
            else:
                tally["WRONG"] += 1
                wrong.append((b["title"], b["no"], b["year"]))
        print(f"\nAUTO-EVAL {label}: {len(sample)} questions -> {tally}")
        for w in wrong[:5]:
            print("   wrong:", w)
    print("\nselftest:", "ALL FIXED CHECKS PASSED" if not bad else f"{bad} FIXED CHECK(S) FAILED")
