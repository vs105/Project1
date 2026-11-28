# Testing

Standard `unittest` (Python standard library - nothing extra to install).
The battery is organised as a pyramid: fast narrow tests at the bottom,
whole-system and full-data checks at the top.

| Level | File | What it proves | Run |
|---|---|---|---|
| Unit | `trial_selftest.py` (37 tests) | attribution rules, identity resolution, follow-up memory, answer shape | `python trial_selftest.py` |
| Unit | `bills_db_selftest.py` | the metadata table in isolation | `python bills_db_selftest.py` |
| Contract | `web_ui_contract_selftest.py` (7 tests) | the UI/engine interface cannot drift | `python web_ui_contract_selftest.py` |
| Integration | `qa_suite.py` - levels 1-3 | question -> answer + answer path, for every phrasing family | `python qa_suite.py` |
| System | `qa_suite.py` - system level | `start_trial.py --check` runs green in a real subprocess; the report generator produces every section | `python qa_suite.py` |
| Adversarial | `qa_suite.py` - adversarial level | refuses to invent: out-of-range years, Acts vs bills, unknown acronyms, false premises, SQL/prompt injection | `python qa_suite.py` |
| Consistency | `qa_suite.py` - consistency level | hundreds of real CSV rows re-asked; every answer must match its row | `python qa_suite.py` |
| Acceptance (full data) | `bill_audit.py` | all 5,169 bills x 4 questions (identity / who / status / when), checked against the CSV; writes `BILL-AUDIT.txt` + `BILL-AUDIT-ALL.csv` | `python bill_audit.py` |
| Data audit | `make_report.py` | files, metadata table, index coverage, quarantine; writes a dated report + `brain-coverage.csv` | `python make_report.py --root . --out _report` |

One command for the everyday gate:

```powershell
.\.venv\Scripts\python.exe .\start_trial.py --check
```

`qa_suite.py --fast` is the seconds-long version of the pyramid's middle.

## Latest results (on the full 5,169-bill table)

- unit + contract: 37 + 7 tests, all pass
- auto-evaluation: 300/300 correct by bill number; 283 correct, 17 clarified,
  0 wrong by title
- qa_suite: all ten levels pass, fast and full mode
- acceptance: **5,169 of 5,169 bills pass all four questions** (13 status and
  1 date check skip where Sansad's own field is blank)

## What "correct" means here

An answer is correct only when it agrees with the metadata row (or the bill
document) it cites. A clarification question ("which of these two bills?")
counts as correct behaviour, not a failure; a confident wrong answer is the
only true failure, and the suites are built to catch exactly that.
