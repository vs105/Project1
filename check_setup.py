#!/usr/bin/env python3
"""
Where am I? One command that says what is built and what to run next.

Run this whenever you are unsure, for example after closing the terminal or
running "deactivate". It changes nothing, it only reports.

    python check_setup.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def tick(ok):
    return "YES" if ok else "no "


def main():
    print("=" * 66)
    print("WHERE AM I?  (this only reports, it changes nothing)")
    print("=" * 66)
    print("folder: %s" % HERE)
    print()

    # ---- 1. the virtual environment -------------------------------------
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    venv_dir = os.path.join(HERE, ".venv")
    print("1. Virtual environment")
    print("   .venv folder exists ............ %s" % tick(os.path.isdir(venv_dir)))
    print("   you are INSIDE it right now .... %s" % tick(in_venv))
    mk = "py -m venv .venv" if os.name == "nt" else "python3 -m venv .venv"
    act = (".\\.venv\\Scripts\\Activate.ps1" if os.name == "nt"
           else "source .venv/bin/activate")
    if os.path.isdir(venv_dir) and not in_venv:
        print()
        print("   >>> Nothing is broken. You just left it. Go back in with:")
        print("       %s" % act)
        print()
    if not os.path.isdir(venv_dir):
        print("   >>> Not made yet. Make it with:")
        print("       %s" % mk)
        print("       %s" % act)
        print()

    # ---- 2. the packages -------------------------------------------------
    print("2. Packages")
    missing = []
    for mod in ("torch", "fitz", "chromadb", "sentence_transformers",
                "flask", "rank_bm25", "pandas", "requests", "tqdm"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    print("   all importable ................. %s" % tick(not missing))
    if missing:
        print("   missing: %s" % ", ".join(missing))
        print("   >>> Install with:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("       pip install -r requirements.txt")
        print()

    # ---- 3. the bills ----------------------------------------------------
    print("3. The bills")
    for cand in ("bills_data", os.path.join(HERE, "..", "bills_data")):
        if os.path.isdir(cand):
            break
    csv_path = os.environ.get("BILLS_CSV") or os.path.join(
        cand, "bills_metadata.csv") if cand else None
    pdfs = 0
    if cand and os.path.isdir(cand):
        for root, _, files in os.walk(cand):
            pdfs += sum(1 for f in files if f.lower().endswith(".pdf"))
    print("   bills_data folder .............. %s" % tick(bool(cand and os.path.isdir(cand))))
    print("   metadata csv ................... %s" % tick(bool(csv_path and os.path.isfile(csv_path))))
    print("   PDFs downloaded ................ %d" % pdfs)
    print("   BILLS_CSV is set ............... %s" % tick(bool(os.environ.get("BILLS_CSV"))))
    if not (csv_path and os.path.isfile(csv_path)):
        print("   >>> Download with:  python download_bills.py")
        print()

    # ---- 4. clean chunks -------------------------------------------------
    print("4. Clean text")
    chunks = os.path.join(HERE, "clean_data", "chunks.json")
    n = 0
    if os.path.isfile(chunks):
        import json
        try:
            n = len(json.load(open(chunks, encoding="utf-8")))
        except Exception:
            n = -1
    print("   clean_data/chunks.json ......... %s%s"
          % (tick(os.path.isfile(chunks)), "  (%d chunks)" % n if n > 0 else ""))
    if not os.path.isfile(chunks):
        print("   >>> Build with:  python step2_organize_clean.py --in bills_data --out clean_data")
        print()

    # ---- 5. the brain ----------------------------------------------------
    print("5. The brain")
    brain = os.path.join(HERE, "brain")
    has_brain = os.path.isfile(os.path.join(brain, "embeddings.npy"))
    print("   brain/embeddings.npy ........... %s" % tick(has_brain))
    if not has_brain:
        print("   >>> Build with:  python step3_build_brain.py --data clean_data/chunks.json --out brain")
        print()

    # ---- 6. the key ------------------------------------------------------
    print("6. Groq API key")
    key = os.environ.get("GROQ_API_KEY", "")
    print("   GROQ_API_KEY is set ............ %s" % tick(bool(key)))
    if not key:
        print("   Not needed to start. Paste it into the box on the web page,")
        print("   or set it with:  $env:GROQ_API_KEY = \"gsk_...\"")
    print()

    # ---- the one next command -------------------------------------------
    print("=" * 66)
    if not in_venv and os.path.isdir(venv_dir):
        print("NEXT: %s" % act)
        print("      then run:  python check_setup.py   again")
    elif not os.path.isdir(venv_dir):
        print("NEXT: %s   (then activate it)" % mk)
    elif missing:
        print("NEXT: pip install -r requirements.txt")
    elif not os.path.isfile(csv_path or ""):
        print("NEXT: python download_bills.py")
    elif not os.path.isfile(chunks):
        print("NEXT: python step2_organize_clean.py --in bills_data --out clean_data")
    elif not has_brain:
        print("NEXT: python step3_build_brain.py --data clean_data/chunks.json --out brain")
    else:
        print("NEXT: python server.py --port 8001    then open http://localhost:8001")
    print("=" * 66)


if __name__ == "__main__":
    main()
