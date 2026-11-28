# 5. clean_data

| file | size | rows |
|---|---:|---:|
| `chunks.csv` | 130.4 MB | 2299173 |
| `chunks.json` | 159.2 MB | 101801 |
| `quarantine.csv` | 129.8 KB | 965 |

## Quarantined files (step 2 could not parse them)

- rows: **964**

| reason | files |
|---|---:|
| NEEDS_OCR | 877 |
| NOT_ENGLISH | 87 |

Note: `chunks.csv` is a human-readable copy of `chunks.json`. No step reads it; it can be deleted to save space without affecting the chatbot.
