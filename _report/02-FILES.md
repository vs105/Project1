# 2. Files

Folder totals. Big trees (`brain/`, `bills_data/pdfs/`, `.venv/`) are summed,
never listed file by file.

| folder | files | size |
|---|---:|---:|
| `bills_data (incl. subfolders)` | 7,861 | 1.8 GB |
| `brain (incl. subfolders)` | 18 | 1.5 GB |
| `brain` | 7 | 387.4 MB |
| `clean_data` | 3 | 289.8 MB |
| `clean_data (incl. subfolders)` | 3 | 289.8 MB |
| `bills_data` | 4 | 9.0 MB |
| `(root)` | 20 | 1.2 MB |
| `docs` | 3 | 6.4 KB |
| `ui` | 1 | 6.0 KB |
| `.streamlit` | 1 | 219 B |

## Code and document files (with hashes)

Hashes let you tell whether a file was edited after this report was made.

| file | size | modified | sha256 (first 16) |
|---|---:|---|---|
| `.streamlit\config.toml` | 219 B | 2026-11-26 21:24 | `f9cfe70a16ed5801` |
| `BILL-AUDIT-ALL.csv` | 758.9 KB | 2026-11-27 02:43 | `1c1b2ff8146289f9` |
| `BILL-AUDIT.txt` | 1.2 KB | 2026-11-27 02:43 | `28e8141738bbb7aa` |
| `README.md` | 2.5 KB | 2026-11-26 21:25 | `d52f5cc7b9e8acc7` |
| `app.py` | 7.8 KB | 2026-11-26 21:24 | `fa0344eb93d6c1ec` |
| `attribution.py` | 12.6 KB | 2026-11-26 20:05 | `dd43568a878d708a` |
| `bill_audit.py` | 12.5 KB | 2026-11-26 20:52 | `b5cdc26d9fc9749d` |
| `bills_data\bills_metadata.csv` | 1.9 MB | 2026-10-17 19:44 | `2f2773640b983243` |
| `bills_data\bills_metadata.json` | 5.4 MB | 2026-10-17 19:44 | `ca82a3ced277789f` |
| `bills_data\bills_metadata_merge.csv` | 149.0 KB | 2026-10-17 19:44 | `1f4eaff128b02663` |
| `bills_data\download_log.csv` | 1.6 MB | 2026-10-17 21:05 | `50c8ed9e61a928e4` |
| `bills_db.py` | 61.0 KB | 2026-11-26 20:05 | `826aa4f33063aa2c` |
| `bills_db_selftest.py` | 2.6 KB | 2026-11-26 20:05 | `22ced6dab1288e61` |
| `clean_data\quarantine.csv` | 129.8 KB | 2026-10-17 22:14 | `eba7fbca4dd097d8` |
| `docs\DATA.md` | 2.0 KB | 2026-11-26 21:26 | `e0bac359c59119f9` |
| `docs\TESTING.md` | 2.5 KB | 2026-11-26 21:26 | `4baeca3c558be58c` |
| `docs\USER-GUIDE.md` | 1.9 KB | 2026-11-26 21:26 | `a7bee8b15d252687` |
| `download_bills.py` | 30.3 KB | 2026-11-26 20:05 | `e819eea2d7288937` |
| `make_report.py` | 46.8 KB | 2026-11-26 20:05 | `0f4633250206e2b2` |
| `qa_suite.py` | 35.5 KB | 2026-11-26 21:24 | `20e845023e8587d0` |
| `requirements.txt` | 638 B | 2026-11-26 20:05 | `453574524f4f0ab8` |
| `server.py` | 12.7 KB | 2026-11-26 20:05 | `2e8bd6a90f6fe376` |
| `start_trial.py` | 6.3 KB | 2026-11-26 21:25 | `af1d00564957abdf` |
| `step2_organize_clean.py` | 30.5 KB | 2026-11-26 20:05 | `6ba25eabc364876f` |
| `step3_build_brain.py` | 39.0 KB | 2026-11-26 20:05 | `de6f6128baaf8cd4` |
| `step4_talk.py` | 122.7 KB | 2026-11-26 20:05 | `5c547196e070149e` |
| `trial_selftest.py` | 19.6 KB | 2026-11-26 20:05 | `dcfdf0551cb1459a` |
| `ui\index.html` | 6.0 KB | 2026-11-26 20:05 | `6fef00817d1df1ea` |
| `web_ui_contract_selftest.py` | 4.5 KB | 2026-11-26 20:05 | `fca30e029c9be9b1` |

## Which generation of the code is in the project root

- `step4_talk.py` looks like: **V2 (patched trial)**
- `attribution.py` present: **yes** (required by the new code, absent from V1)
- `trial_selftest.py` present: **yes**
- `start_trial.py` present: **yes**
- `v2_trial/` folder present: **no**

If this says V1 while you expected V2, the new files are in a subfolder (or were copied to the wrong place) - that single line explains most 'nothing changed' reports.

## Python files in the root, by name

This is the answer to 'which version of step4_talk.py am I running?'.

- `app.py` - 7.8 KB, 2026-11-26 21:24
- `attribution.py` - 12.6 KB, 2026-11-26 20:05
- `bill_audit.py` - 12.5 KB, 2026-11-26 20:52
- `bills_db.py` - 61.0 KB, 2026-11-26 20:05
- `bills_db_selftest.py` - 2.6 KB, 2026-11-26 20:05
- `download_bills.py` - 30.3 KB, 2026-11-26 20:05
- `make_report.py` - 46.8 KB, 2026-11-26 20:05
- `qa_suite.py` - 35.5 KB, 2026-11-26 21:24
- `server.py` - 12.7 KB, 2026-11-26 20:05
- `start_trial.py` - 6.3 KB, 2026-11-26 21:25
- `step2_organize_clean.py` - 30.5 KB, 2026-11-26 20:05
- `step3_build_brain.py` - 39.0 KB, 2026-11-26 20:05
- `step4_talk.py` - 122.7 KB, 2026-11-26 20:05
- `trial_selftest.py` - 19.6 KB, 2026-11-26 20:05
- `web_ui_contract_selftest.py` - 4.5 KB, 2026-11-26 20:05
