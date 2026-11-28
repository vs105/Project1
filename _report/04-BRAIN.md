# 4. Brain (the search index)

## Files

| file | size |
|---|---:|
| `bm25.pkl` | 79.9 MB |
| `cache_ids.json` | 2.9 MB |
| `cache_model.txt` | 38 B |
| `chroma_db/` (11 files) | 1.1 GB |
| `embeddings.npy` | 146.5 MB |
| `fingerprint.json` | 324 B |
| `kept_chunks.json` | 158.1 MB |
| `retrieval_demo.txt` | 3.7 KB |

## `cache_model.txt`

```text
sentence-transformers/all-MiniLM-L6-v2
```

## `fingerprint.json`

```text
{"count": 99997, "first": "17_2008_billGazetted_1804", "last": "XLI_2026_billPassedIn_101800", "model": "sentence-transformers/all-MiniLM-L6-v2", "meta": ["bill_no", "bill_no_norm", "bill_title", "year", "term", "ministry", "bill_type", "status", "slot", "language", "part", "clause_no", "label", "house", "source_primary"]}
```

## Size and shape

- chunks in the index: **99,997**
- distinct bills covered: **4,589**
- years covered: **23**
- chunks with empty text: **0**
- chunks whose text is an exact duplicate of another: **5,257**
- text length: shortest 34, median 590, longest 117,636 characters

## By `house`

| value | chunks |
|---|---:|
| `Lok Sabha` | 77,562 |
| `Rajya Sabha` | 22,435 |

## By `part`

| value | chunks |
|---|---:|
| `operative` | 84,579 |
| `narrative` | 15,418 |

## By `slot`

| value | chunks |
|---|---:|
| `billIntroducedFile` | 77,827 |
| `billGazettedFile` | 11,367 |
| `billPassedInBothHousesFile` | 6,538 |
| `billPassedInLSFile` | 3,581 |
| `billPassedInRSFile` | 684 |

## By `language`

| value | chunks |
|---|---:|
| `english` | 99,997 |

## By `source_primary`

| value | chunks |
|---|---:|
| `sansad` | 99,997 |

## Fields the talk layer relies on

- chunks with a clause number: **84,561** of 99,997
- chunks with `bill_no_norm`: **99,997** (0 means the brain predates that column and falls back to raw numbers)

## Coverage - the important number

- bills in the metadata table: **5,169**
- bills with at least one chunk of text: **4,589** (88.8%)
- bills with **no text at all**: **580**

A question about a bill in the last row of that list can only be answered from metadata (dates, status, introducer), never from its clauses. That is a data limit, not a code bug - and it is listed in `brain-coverage.csv` so each bill can be checked by name.

## Chroma database (read-only, via sqlite)

- tables: `migrations`, `acquire_write`, `collection_metadata`, `segment_metadata`, `tenants`, `databases`, `collections`, `maintenance_log`, `segments`, `embeddings`, `embedding_metadata`, `max_seq_id`
- collection `bills` = `c26a84fb-7b0b-437e-98ff-94d2340a22bd`
- rows in `embeddings`: **99,997**
- chunks in kept_chunks.json: **99,997**

## Files written next to this one

- `brain-coverage.csv` - one row per bill: how many chunks of text the index holds for it, and `has_text` yes/no.

