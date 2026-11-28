# 9. Questions asked of this project's own code

Each row is the real answer function in the project being inspected, run with no API key and no search index, so nothing is downloaded and no model is called. The path is the line the app shows under an answer.

| question | why this probe | answer path |
|---|---|---|
| `Who introduced Lok Sabha Bill No. 10 of 2004?` | oldest bill in the table | `bill-metadata` |
| `Who introduced Rajya Sabha Bill No. XXXVII of 2026?` | newest bill in the table | `bill-metadata` |
| `Who introduced Rajya Sabha Bill No. XXXVII of 2026?` | a Rajya Sabha bill | `bill-metadata` |
| `Who introduced Lok Sabha Bill No. 280 of 2016?` | a Lok Sabha bill, mid-table | `bill-metadata` |
| `Who introduced Lok Sabha Bill No. 100 of 2004?` | a row with no introducer recorded | `bill-metadata` |
| `Tell me about The Constitution (Amendment) Bill,2016 (Amendment of article 15)` | a bill named by its title | `clarify-bill` |
| `What does The Constitution (Amendment) Bill,2016 (Amendment of article 15) say?` | a text question with no index | `clarify-bill` |
| `Who proposed this bill?` | a follow-up with no bill yet named | `clarify-bill` |
| `What is the capital of France?` | off-topic | `facts-mode` |
| `Tell me about Bill No. 999 of 2019.` | a bill number that does not exist | `no-such-bill` |
| `Tell me about Bill No. 370 of 1999.` | a year outside 2004-2026 | `no-such-bill` |
| `How many bills were introduced in 2019?` | a count | `bill-metadata` |
| `Which bills were passed in December 2019?` | a date list | `bill-metadata` |
| `Hello` | greeting | `conversation` |

## Full text of those answers

**Q:** Who introduced Lok Sabha Bill No. 10 of 2004?  
*oldest bill in the table · path `bill-metadata`*

```text
**The Prevention of Terrorism (Repeal) Bill, 2004** (Bill No. 10 of 2004, Lok Sabha): The metadata records the introducer as **C.K. Chandrappan Shri**. _Source: Sansad bill metadata (primary/official)._
```

**Q:** Who introduced Rajya Sabha Bill No. XXXVII of 2026?  
*newest bill in the table · path `bill-metadata`*

```text
**The Indian Diaspora and Migrant Workers (Protection and Welfare) Bill, 2026** (Bill No. XXXVII of 2026, Rajya Sabha): The metadata records the introducer as **Ashok Kumar Mittal Dr.** _Source: Sansad bill metadata (primary/official)._
```

**Q:** Who introduced Rajya Sabha Bill No. XXXVII of 2026?  
*a Rajya Sabha bill · path `bill-metadata`*

```text
**The Indian Diaspora and Migrant Workers (Protection and Welfare) Bill, 2026** (Bill No. XXXVII of 2026, Rajya Sabha): The metadata records the introducer as **Ashok Kumar Mittal Dr.** _Source: Sansad bill metadata (primary/official)._
```

**Q:** Who introduced Lok Sabha Bill No. 280 of 2016?  
*a Lok Sabha bill, mid-table · path `bill-metadata`*

```text
**The Cigarettes and Other Tobacco Products (Prohibition of Advertisement and Regulation of Trade and Commerce, Production, Supply and Distribution) Amendment Bill, 2016 (Amendment of sections 4 and 6)** (Bill No. 280 of 2016, Lok Sabha): The metadata records 
```

**Q:** Who introduced Lok Sabha Bill No. 100 of 2004?  
*a row with no introducer recorded · path `bill-metadata`*

```text
**The Appropriation (Railways) No.4 Bill, 2004** (Bill No. 100 of 2004, Lok Sabha): No individual introducer is named in this metadata row. The recorded ministry is RAILWAYS. A ministry alone does not identify the person who moved the bill. _Source: Sansad bil
```

**Q:** Tell me about The Constitution (Amendment) Bill,2016 (Amendment of article 15)  
*a bill named by its title · path `clarify-bill`*

```text
I found 5 bills that match, so I won't guess: - The Constitution (Amendment) Bill,2016 (Amendment of article 15) - Bill No. 155 of 2016 (Private Member, Lapsed) - The Constitution (Amendment) Bill,2016 (Amendment of article 15) - Bill No. 173 of 2016 (Private 
```

**Q:** What does The Constitution (Amendment) Bill,2016 (Amendment of article 15) say?  
*a text question with no index · path `clarify-bill`*

```text
I found 5 bills that match, so I won't guess: - The Constitution (Amendment) Bill,2016 (Amendment of article 15) - Bill No. 155 of 2016 (Private Member, Lapsed) - The Constitution (Amendment) Bill,2016 (Amendment of article 15) - Bill No. 173 of 2016 (Private 
```

**Q:** Who proposed this bill?  
*a follow-up with no bill yet named · path `clarify-bill`*

```text
Which bill do you mean? I couldn't identify one matching record. Please give the House, bill number and year, or the full title and year.
```

**Q:** What is the capital of France?  
*off-topic · path `facts-mode`*

```text
I could not identify a bill in that question, so I will not guess. I hold 5,169 bills from 2004 to 2026, Lok Sabha and Rajya Sabha, current to 2026-08-10. Give a bill title, or the House with the bill number and year - for example "Rajya Sabha Bill No. II of 2
```

**Q:** Tell me about Bill No. 999 of 2019.  
*a bill number that does not exist · path `no-such-bill`*

```text
No bill numbered 999 of 2019 is in this data. I hold 5,169 bills from 2004 to 2026, in both Houses, current to 2026-08-10. Please check the number, the year and the House. If that bill does exist, then the gap is in my copy of the data, not in your question.
```

**Q:** Tell me about Bill No. 370 of 1999.  
*a year outside 2004-2026 · path `no-such-bill`*

```text
No bill numbered 370 of 1999 is in this data. 1999 is outside the 2004-2026 range of this data. I hold 5,169 bills from 2004 to 2026, in both Houses, current to 2026-08-10. Please check the number, the year and the House. If that bill does exist, then the gap 
```

**Q:** How many bills were introduced in 2019?  
*a count · path `bill-metadata`*

```text
**420 bills** match (2019; counted by introduction year). _Metadata provenance: 5169 Sansad-primary records. Verify bill details against Sansad._
```

**Q:** Which bills were passed in December 2019?  
*a date list · path `bill-metadata`*

```text
**8 bills** match (month 12; 2019; by Lok Sabha passing date). - The Recycling of Ships Bill, 2019 - Bill No. 361 of 2019 (Lok Sabha, Assented) - The Taxation Laws (Amendment) Bill, 2019 - Bill No. 362 of 2019 (Lok Sabha, Assented) - The International Financia
```

**Q:** Hello  
*greeting · path `conversation`*

```text
Hello. Ask me anything about Indian Parliament bills from 2004 to 2026 - Lok Sabha and Rajya Sabha both. You can name a bill, ask what it does, ask when it was passed, or ask how many bills there were in a year.
```

