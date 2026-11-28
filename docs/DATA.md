# Data

## Source

All data comes from the official Sansad portal (bills API + published bill
PDFs/DOCs), downloaded 2026-08-10. The metadata table is a snapshot, not a
live feed; bills introduced after that day are not in this copy.

## The table

`bills_data/bills_metadata.csv` - 5,169 bills, 29 columns.

| split | count |
|---|---:|
| Lok Sabha | 3,678 |
| Rajya Sabha | 1,491 |
| Government bills | 1,165 |
| Private Member bills | 4,004 |
| Ordinary / Financial / Money / Constitution Amendment / Ordinance-replacing | 2,551 / 2,066 / 308 / 231 / 13 |

Known blanks in the source itself: 1,163 rows have no individual introducer
recorded (a ministry is recorded instead), 13 have no status, 1 has no
introduction date. The app reports these blanks rather than filling them.

## The documents

7,857 files - more files than bills because one bill can have several
documents:

| stage | files |
|---|---:|
| introduced | 4,837 |
| errata | 888 |
| passed both Houses | 654 |
| gazette | 548 |
| passed Lok Sabha | 509 |
| synopsis | 312 |
| passed Rajya Sabha | 109 |

Extensions: 7,031 `.pdf`, 413 `.doc`, 413 `.txt`.

## The index ("brain")

`brain/` holds 99,997 text chunks (model `all-MiniLM-L6-v2`), covering
**4,589 of 5,169 bills (88.8%)**.

**580 bills have no readable text** (453 Private Member / 127 Government;
367 Lok Sabha / 213 Rajya Sabha). Reasons, both data-side:

- no document was published/downloaded for the bill, or
- the published PDF is a scan - 877 files are quarantined as NEEDS_OCR
  (plus 87 NOT_ENGLISH) during cleaning and never reach the index.

For those 580 bills the app answers metadata questions (dates, status,
introducer) and says plainly that clause text is unavailable.

## Regenerating the audits

```powershell
.\.venv\Scripts\python.exe .\make_report.py --root . --out _report
.\.venv\Scripts\python.exe .\bill_audit.py
```

`make_report.py` writes a dated report folder plus `brain-coverage.csv`
(per-bill chunk counts); `bill_audit.py` writes `BILL-AUDIT.txt` and
`BILL-AUDIT-ALL.csv` (per-bill answer verdicts). Both are read-only.
