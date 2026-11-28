"""Start a side-by-side trial from the existing bills-chatbot project. No installs or rebuilds."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def _smoke():
    """Ask the real CSV the questions this app exists to get right."""
    from step4_talk import FactsMode, get_facts, answer_question
    db = get_facts()
    if db is None:
        raise SystemExit('The metadata CSV could not be loaded.')
    brain = FactsMode()
    ctx = None
    checks = [
        ('Which MP introduced Rajya Sabha Bill No. II of 2023?', 'John Brittas'),
        ('Who proposed this bill?', 'John Brittas'),
        ('Who proposed Rajya Sabha Bill No. XLIII of 2016?', 'Source conflict'),
        ('Who proposed Lok Sabha Bill No. 370 of 2019?', None),
        ('Who proposed Rajya Sabha Bill No. 370 of 2019?', 'No Rajya Sabha bill'),
        ('When was Bill No. 370 of 2019 passed?', 'Passed by Lok Sabha'),
        ('How many private member bills were introduced?', '4004 bills'),
        ('What does the DPDP Bill say?', 'Document search is off'),
    ]
    for question, expected in checks:
        reply, _hits, used = answer_question(brain, question, last_filters=ctx)
        ctx = brain.last_filters
        flat = ' '.join(reply.split())
        if expected is None:
            # No name in the metadata; either the introduced PDF names the
            # signatory (good) or the answer says no individual is named.
            ok = ('signatory' in flat) or ('No individual introducer' in flat)
        else:
            ok = expected.casefold() in flat.casefold()
        print(('PASS ' if ok else 'FAIL ') + question + f'  [{used}]  ->  {flat[:150]}')
        if not ok:
            raise SystemExit('A live question gave an unexpected answer; nothing was changed on disk.')
    print(f'{len(checks)} live questions answered as expected, from {len(db.bills)} metadata rows.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true',help='run the offline metadata and integration tests, then exit')
    ap.add_argument('--port',type=int,default=8502)
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    # TWO SHAPES ARE SUPPORTED, because both are reasonable:
    #   nested  - the code sits in bills-chatbot\v2_trial\ next to
    #             bills_data\ (the way the ZIP arrives)
    #   merged  - the code IS the project folder, with bills_data\, brain\ and
    #             clean_data\ beside it (the way you run it once you trust it)
    # The CSV location decides which one you are in, so no flag is needed.
    candidates=[here.parent/'bills_data'/'bills_metadata.csv',
                here/'bills_data'/'bills_metadata.csv']
    csv_path=next((c for c in candidates if c.is_file()),None)
    if csv_path is None:
        raise SystemExit(
            f'No bills_metadata.csv found. Looked in:\n  '
            + '\n  '.join(str(c) for c in candidates)
            + '\n\nEither keep bills_data\\ next to this folder (nested layout), or put\n'
              'these files directly in the project folder beside bills_data\\(merged layout).\n'
              'Do not move the data itself.')
    root=csv_path.parent.parent
    os.chdir(root)
    sys.path.insert(0,str(here))
    os.environ['BILLS_CSV']=str(csv_path)
    os.environ.setdefault('PYTHONUTF8','1')
    print(f'Code: {here}\nCSV: {csv_path}\nPython: {sys.executable}',flush=True)
    if args.check:
        for file,params in [('trial_selftest.py',[]),('web_ui_contract_selftest.py',[]),('bills_db.py',['--csv',str(csv_path),'--selftest'])]:
            print(f'\n--- {file} ---',flush=True)
            res=subprocess.run([sys.executable,'-B',str(here/file),*params]);
            if res.returncode: raise SystemExit(res.returncode)
        print('\n--- your data: eight live questions ---',flush=True)
        _smoke()
        print('\nAll checks passed. No PDFs were downloaded and the real brain was not opened.'); return
    try:
        from streamlit.web import cli
    except ImportError as e:
        # A venv that never had Streamlit is a common, fixable mistake. Say
        # exactly what is missing, which interpreter is running, and the one
        # command that fixes it - instead of a bare ImportError.
        raise SystemExit(
            f'Streamlit is not installed in the Python that is running this file:\n'
            f'  {sys.executable}\n'
            f'Reason: {e}\n\n'
            f'Two ways to fix it:\n'
            f'  1. Use the project .venv (recommended):\n'
            f'     & ".\\.venv\\Scripts\\python.exe" ".\\v2_trial\\start_trial.py"\n'
            f'  2. Or install Streamlit into this interpreter:\n'
            f'     "{sys.executable}" -m pip install streamlit\n')
    if not 1 <= args.port <= 65535: raise SystemExit('Choose a port from 1 to 65535.')
    # A Streamlit install that reports development mode refuses --server.port,
    # which would otherwise end in a confusing "developmentMode" traceback.
    # This app is never served in development mode, so switch it off.
    if os.environ.get('STREAMLIT_GLOBAL_DEVELOPMENT_MODE', '').lower() == 'true':
        print('Note: Streamlit development mode was switched off so --port works.', flush=True)
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    # Tell the person what they are looking at before the browser opens, so a
    # wrong folder or a missing index is obvious from the terminal alone.
    csv_used = os.environ.get('BILLS_CSV', '(not set)')
    brain_dir = root / 'brain'
    print(f'Project folder : {root}', flush=True)
    print(f'Code folder    : {here}', flush=True)
    print(f'Metadata CSV   : {csv_used}', flush=True)
    if (brain_dir / 'kept_chunks.json').is_file():
        print(f'Brain          : found at {brain_dir} '
              f'(tick "Enable document search" in the sidebar to use it)', flush=True)
    else:
        print(f'Brain          : NOT found at {brain_dir} - clause questions will say '
              f'document search is unavailable. Metadata questions still work.', flush=True)
    print(f'Open http://127.0.0.1:{args.port} (Ctrl+C stops the app).', flush=True)
    sys.argv=['streamlit','run',str(here/'app.py'),'--server.address=127.0.0.1',
              f'--server.port={args.port}','--server.headless=true',
              '--browser.gatherUsageStats=false','--server.fileWatcherType=none']
    raise SystemExit(cli.main())

if __name__=='__main__': main()
