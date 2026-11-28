# 3. Metadata table

- file: `bills_data\bills_metadata.csv`
- size: 1.9 MB
- rows: **5,169**
- columns: **29**
- sha256: `2f2773640b9832431dbe06b6c1f44b46ac0de1df871aed540944bf5f5133819d`

## Columns, with blanks

| column | blank / missing |
|---|---:|
| `house` | 0 |
| `billNumber` | 0 |
| `bill_number_norm` | 0 |
| `billYear` | 0 |
| `billName` | 0 |
| `billType` | 0 |
| `billCategory` | 0 |
| `ministryName` | 142 |
| `billIntroducedInHouse` | 0 |
| `billIntroducedBy` | 1,163 |
| `billIntroducedDate` | 1 |
| `billPassedInLSDate` | 4,254 |
| `billPassedInRSDate` | 4,321 |
| `referredToCommitteeDate` | 5,061 |
| `reportPresentedDate` | 5,070 |
| `actNo` | 4,329 |
| `actYear` | 4,329 |
| `billAssentedDate` | 4,298 |
| `status` | 13 |
| `billIntroducedFile` | 333 |
| `billPassedInLSFile` | 4,663 |
| `billPassedInRSFile` | 5,059 |
| `billPassedInBothHousesFile` | 4,516 |
| `errataFile` | 4,672 |
| `reportFile` | 5,140 |
| `billGazettedFile` | 4,619 |
| `billSynopsisFile` | 4,856 |
| `source_primary` | 0 |
| `filled_from` | 5,168 |

## `house`

| value | rows |
|---|---:|
| Lok Sabha | 3,678 |
| Rajya Sabha | 1,491 |

## `status`

| value | rows |
|---|---:|
| Lapsed | 3,362 |
| Assented | 842 |
| Pending | 524 |
| Removed | 222 |
| Withdrawn | 144 |
| Passed | 50 |
| (blank) | 13 |
| Negatived | 12 |

## `billType`

| value | rows |
|---|---:|
| Private Member | 4,004 |
| Government | 1,165 |

## `source_primary`

| value | rows |
|---|---:|
| sansad | 5,169 |

## What the numbers mean

- distinct (house, number, year) keys: **5,169**
- rows repeating an existing key: **0** (0 means one row per bill)
- `billYear` range: **2004-2026**
- rows with `billYear` outside 2004-2026: **0**
- rows with no introducer recorded: **1,163**
- newest `billIntroducedDate`: **2026-08-10** (this is the data's as-of date)
- oldest `billIntroducedDate`: **2003-12-19**
- rows whose title year differs from `billYear`: **161** (normal: a February bill carries the previous year's number)

## First 3 rows, as they are stored

```text
{"house": "Lok Sabha", "billNumber": "10", "billYear": "2004", "billName": "The Prevention of Terrorism (Repeal) Bill, 2004", "status": "Pending", "billIntroducedBy": "C.K. Chandrappan Shri", "billIntroducedDate": "2004-09-07 00:00:00.0"}
{"house": "Lok Sabha", "billNumber": "100", "billYear": "2004", "billName": "The Appropriation (Railways) No.4 Bill, 2004", "status": "Assented", "billIntroducedBy": "", "billIntroducedDate": "2004-12-08 00:00:00.0"}
{"house": "Lok Sabha", "billNumber": "101", "billYear": "2004", "billName": "The Government Securities Bill, 2004", "status": "Assented", "billIntroducedBy": "", "billIntroducedDate": "2004-12-21 00:00:00.0"}
```

