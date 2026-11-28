"""Read-only introducer evidence. No downloads, model calls, OCR or index writes.
CSV 'introduced by' and a signature are different kinds of evidence.
"""
import bisect
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re

EXTRACTOR_MISSING = False  # set when a matching PDF exists but PyMuPDF cannot be imported

PERSON_QUERY = re.compile(
    r'\b(?:who|whom|which\s+(?:mp|minister|member(?:\s+of\s+parliament)?|person))\b[^?\n]*?\b(?:propos|introduc|mov(?:e|er)|sponsor|tabl(?:e|ed)|brought)\w*\b'
    r'|\b(?:introduc|propos|sponsor|mov(?:e|er))\w*\b[^?\n]*?\b(?:who|whom|which\s+(?:mp|minister|member))\b'
    r'|\b(?:who\s+is\s+the\s+author|(?:name|identity)\s+of\s+the\s+(?:mover|sponsor|introducer)|how\s+(?:was|is|were)\b[^?\n]*?\bproposed|(?:mover|sponsor|introducer)\s+of)\b', re.I)

def asks_person(q):
    return bool(PERSON_QUERY.search(q or ''))


def conflict_note(b):
    # Reviewed exception, not a change to the raw CSV. Match title AND key AND
    # the conflicting value so a future corrected record is not overwritten.
    if (str(b.get('house','')).casefold() == 'rajya sabha' and str(b.get('norm')) == '43'
            and str(b.get('year')) == '2016' and 'maternity benefit' in (b.get('title') or '').casefold()
            and 'bangaru laxman' in (b.get('by') or '').casefold()):
        return ('Source conflict, not a settled fact.\n\n'
                f'- This CSV records **{b["by"]}** and ends the title in **2017**.\n'
                '- The official introduced bill is the **Maternity Benefit (Amendment) Bill, 2016**; its '
                'Statement of Objects and Reasons is signed **Bandaru Dattatreya**, Minister of State (I/C), '
                'Labour and Employment (PDF page 3).\n'
                '- PRS Legislative Research names Bandaru Dattatreya as the introducer in the Rajya Sabha on '
                '11 August 2016.\n\n'
                'I will not silently pick one of the two names. '
                '[Introduced bill (Sansad)](https://sansad.in/getFile/BillsTexts/RSBillTexts/Asintroduced/maternity-int%2011%208%2016.pdf?source=legislation) '
                '· [PRS bill page](https://prsindia.org/billtrack/the-maternity-benefit-amendment-bill-2016)')
    return ''


def recorded_by(b):
    conflict = conflict_note(b)
    if conflict:
        return conflict
    if b.get('by'):
        name = b['by']
        # "John Brittas Dr." must not become "Dr.." when a full stop is added.
        return 'The metadata records the introducer as **' + name + '**' + ('' if name.endswith('.') else '.')
    ministry = b.get('ministry') or ''
    return ('No individual introducer is named in this metadata row.'
            + (f' The recorded ministry is {ministry}.' if ministry else '')
            + ' A ministry alone does not identify the person who moved the bill.')


# Intentionally conservative. A city/date alone or a signature elsewhere in a
# bill is not evidence of signing its Statement of Objects and Reasons.
SOR = re.compile(r'(?im)^[ \t]*STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS[ \t]*[.:]?[ \t]*$')
END = re.compile(r'(?im)^[ \t]*(?:NOTES\s+ON\s+CLAUSES|FINANCIAL\s+MEMORANDUM|MEMORANDUM\s+(?:REGARDING|ON|OF)|ANNEX(?:URE)?\b|APPENDIX\b|(?:PRESIDENT[’\']S\s+RECOMMENDATION|RECOMMENDATION\s+OF\s+THE\s+PRESIDENT)|(?:LOK|RAJYA)\s+SABHA\s*$)')
LOCATION = re.compile(r'(?im)^[ \t]*NEW\s+DELHI[ \t]*[;,.:]?[ \t]*')
DATE = re.compile(r'\b(?:The\s+)?(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s*,?\s*(?:19|20)\d\d\b[.;]?',re.I)
BAD = set('NEW DELHI LOK RAJYA SABHA PARLIAMENT GOVERNMENT INDIA MINISTER MINISTRY HOUSE BILL ACT SECTION CHAPTER MEMORANDUM ANNEX ANNEXURE APPENDIX REASONS OBJECTS STATEMENT PRESIDENT SECRETARIAT SECRETARY SCHEDULE CLAUSES NOTES FINANCIAL DATED DATE NOTIFICATION CHAIRMAN THE SIGNATURE PRINTED PUBLISHED'.split())
NAME = re.compile(r"[A-Z][A-Z.'’\-]*(?:[ \t]+[A-Z][A-Z.'’\-]*){1,5}\.?\Z")

def pairs(text):
    """Yield (name, literal location/name/date block, name offset). No guessing outside the section."""
    text = (text or '').replace('\r', '')
    for header in SOR.finditer(text):
        end = END.search(text, header.end()); stop = end.start() if end else len(text)
        for city in LOCATION.finditer(text, header.end(), stop):
            span = text[city.end():min(stop, city.end()+175)]
            date = DATE.search(span)
            if not date or date.start()>85: continue
            # Common PDFs put the name before the date; others put it just after.
            pre = span[:date.start()].strip(' \n\t,;.'); post = span[date.end():].lstrip(' \n\t,;.')
            if pre:
                if '\n' in pre: continue
                raw=pre; rel=span.find(pre); block_end=city.end()+date.end()
            else:
                raw=post.split('\n',1)[0].strip() if post else ''; rel=span.find(raw,date.end()); block_end=city.end()+rel+len(raw)
            raw=raw.strip(' \t,;.'); name=' '.join(raw.split()); words=set(re.findall('[A-Z]+',name))
            if not NAME.fullmatch(name) or words & BAD or not any(len(w)>2 for w in words): continue
            yield name, text[city.start():block_end].strip(), city.end()+rel


def _slug(s):
    return re.sub('[^a-z0-9]+','',(s or '').casefold())

@lru_cache(maxsize=8)
def _file_index(pdf_root):
    groups=defaultdict(list)
    for p in Path(pdf_root).glob('*/intro/*'):
        if p.suffix.lower() not in {'.pdf','.doc','.docx','.rtf'}: continue
        fields=p.stem.split('_',3)
        if len(fields)==4 and fields[0].isdigit():
            groups[(fields[1].casefold(),fields[2])].append((p,_slug(fields[3])))
    return groups

@lru_cache(maxsize=192)
def _read(path_str, mtime, size, house, number, year):
    """Only files opened by the local user: no URLs or paths taken from prompts."""
    p=Path(path_str)
    if not 0<size<=160_000_000: return ()
    global EXTRACTOR_MISSING
    try:
        if p.suffix.casefold()=='.pdf':
            try:
                import pymupdf
            except ImportError:
                EXTRACTOR_MISSING = True
                return ()
            with pymupdf.open(str(p)) as doc:
                if doc.needs_pass or len(doc)>1400: return ()
                pages=[page.get_text('text').replace('\r','') for page in doc]
        elif p.suffix.casefold()=='.txt':
            if size>4_000_000: return ()
            pages=[p.read_text(encoding='utf-8-sig',errors='replace').replace('\r','')]
        else: return ()
    except Exception: return ()
    if not pages: return ()
    cover=pages[0]
    h=re.search(r'AS\s+INTRODUCED\s+IN\s+(?:THE\s+)?((?:LOK|RAJYA)\s+SABHA)',cover,re.I)
    if h and ' '.join(h[1].casefold().split())!=house: return ()
    ids=re.findall(r'\bBILL\s+NO\.?\s*([IVXLCDM]+|\d+)\s+OF\s+(\d{4})',cover,re.I)
    if ids and (number.casefold(),year) not in {(n.casefold(),y) for n,y in ids}: return ()
    starts=[]; offset=0
    for pg in pages: starts.append(offset);offset+=len(pg)+1
    text='\n'.join(pages); out=[]
    for name,block,pos in pairs(text):
        pg=bisect.bisect_right(starts,pos)
        ref=f'{p.name}, PDF page {pg}' if p.suffix.casefold()=='.pdf' else f'{p.name} (existing text conversion)'
        out.append((name,block,ref))
    return tuple(out)


def locate(db, question, last_filters):
    """Resolve ONE bill for a person/attribution question, or return a message."""
    res = db.resolve(question, last_filters)
    if res['kind'] == 'many':
        return None, res, db.many(res, question)
    if res['kind'] != 'one':
        mh = re.search(r'\b(lok|rajya)\s+sabha\s+(?:\w+\s+)?bill\s*(?:no\.?|number)\s*([\divxlcdm]{1,8})\s*of\s*(\d{4})', question, re.I)
        if mh:
            _raw, wanted = db.bills[0]['norm'], None
            from bills_db import normalize_bill_number
            _raw, wanted = normalize_bill_number(mh[2])
            other = [b for b in db.bills if b['year'] == mh[3]
                     and (b['norm'] == wanted if wanted is not None
                          else b['no'].casefold() == mh[2].casefold())]
            where = sorted({b['house'] for b in other})
            if where and not any(h.casefold() == mh[1].casefold() + ' sabha' for h in where):
                return None, res, (f'No {mh[1].title()} Sabha bill numbered {mh[2]} in {mh[3]} is in this data, '
                                   f'so I will not substitute another bill. Bill {mh[2]} of {mh[3]} exists in: '
                                   + ', '.join(where) + '.')
        return None, res, ("Which bill do you mean? I couldn't identify one matching record. "
                           "Please give the House, bill number and year, or the full title and year.")
    bill = res['bills'][0]
    mh = re.search(r'\b(lok|rajya)\s+sabha\s+(?:\w+\s+)?bill\b', question, re.I)
    if mh and bill['house'].casefold() != mh[1].casefold() + ' sabha':
        return None, res, 'No matching bill in the specified House. Please check its number and year.'
    # A title and a number that point at different bills in the same number/year
    # group must not be silently settled in favour of one of them.
    from bills_db import QUERY_NOISE
    group = db.raw_groups.get((bill['no'].casefold(), bill['year']), [])
    if len({g['title'] for g in group}) > 1:
        words = set(re.findall(r'[a-z]{4,}', question.casefold())) - QUERY_NOISE
        scores = {g['title']: len(words & g['toks']) for g in group}
        best = max(scores, key=scores.get)
        if scores[best] > scores.get(bill['title'], 0):
            return None, res, ('The title and the number point to different bills. '
                               'Please give one of them so I don\'t guess.')
    return bill, res, ''


def evidence(db,b,brain=None):
    house=str(b.get('house') or '').strip().casefold(); no=str(b.get('no') or ''); year=str(b.get('year') or '')
    if not (house and no and year): return None
    rawkey=(no.casefold(),year); found=[]; matching_files=False
    root=Path(db.csv_path).parent/'pdfs'
    for path,slug in _file_index(str(root)).get(rawkey,[]):
        if not slug: continue
        matches=[x for x in db.raw_groups.get(rawkey,[]) if _slug(x['title']).startswith(slug)]
        if len(matches)!=1 or matches[0] is not b: continue
        if path.suffix.casefold()!='.pdf': path=path.with_suffix('.txt')
        try:
            st=path.stat(); matching_files=True
            found.extend(_read(str(path),st.st_mtime_ns,st.st_size,house,no,year))
        except OSError: continue
    # A confirmed local document takes precedence over potentially stale index text.
    if EXTRACTOR_MISSING and not found:
        return 'EXTRACTOR_MISSING'
    if not matching_files and brain is not None:
        chunks=getattr(brain,'chunks',{})
        for c in chunks.values() if hasattr(chunks,'values') else chunks:
            if c.get('slot')!='billIntroducedFile' or c.get('part')!='narrative': continue
            if (str(c.get('house') or '').strip().casefold(), str(c.get('bill_no') or '').casefold(),str(c.get('year') or ''))!=(house,no.casefold(),year): continue
            if c.get('bill_title') and _slug(c['bill_title'])!=_slug(b['title']): continue
            for name,block,_ in pairs(c.get('text','')):
                found.append((name,block,c.get('source_file') or c.get('chunk_id') or 'indexed introduced document'))
    return found[0] if len({v[0].casefold() for v in found})==1 else None


def response(db,b,brain=None,allow_evidence=True,prefix=''):
    from bills_db import metadata_source_line
    head=prefix+db._head(b)+': '; primary=recorded_by(b)
    if conflict_note(b): return head+primary, 'source-conflict'
    if b.get('by') and not str(b.get('type') or '').casefold().startswith('gov'):
        return head+primary+'\n\n'+metadata_source_line(b), 'bill-metadata'
    ev=evidence(db,b,brain) if allow_evidence else None
    if ev == 'EXTRACTOR_MISSING':
        return (head+primary+'\n\nA matching introduced PDF is in this project, but this Python '
                'cannot read PDFs (PyMuPDF is not installed here), so I will not guess who signed it. '
                'Start the trial with the project .venv to read it.'), 'bill-metadata-no-extractor'
    if not ev:
        return head+primary+'\n\n'+metadata_source_line(b), 'bill-metadata'
    name,block,ref=ev
    first=(primary+' ' if b.get('by') else '')
    return (head+first+'The Statement of Objects and Reasons in the introduced bill is signed by **'+name.title()+'**. '
            'That identifies its signatory; it does not independently establish who moved the motion in the House.'
            +'\n\nSource: '+ref+'\n\n'+ '\n'.join('> '+line for line in block.splitlines() if line.strip())), 'bill-introduction-signature'
