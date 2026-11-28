"""Offline regression tests using fictional bills; the real managed corpus is never written."""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from attribution import pairs
from bills_db import BillsDB
import step4_talk as talk


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.root=root
        self.pdfs=root/'pdfs'/'Y2004-2026'/'intro'; self.pdfs.mkdir(parents=True)
        rows=[('51','The Archive Study Bill, 2024','Government','Example Records',''),
              ('52','The Path Access Bill, 2024','Private Member','','Meera Example'),
              ('53','The Park Study Bill, 2024','Private Member','','')]
        f=root/'bills_metadata.csv'
        with f.open('w',encoding='utf-8',newline='') as out:
            w=csv.writer(out);w.writerow(['billNumber','billName','billType','ministryName','billIntroducedBy','house','billYear','source_primary']);
            for row in rows: w.writerow([*row,'Lok Sabha','2024','sansad'])
        self.db=BillsDB(f);self.brain=talk.FactsMode();override=patch.object(talk,'_DB',self.db);override.start();self.addCleanup(override.stop)
    def ask(self,q):
        return talk.answer_question(self.brain,q,last_filters=self.brain.last_filters)[0]
    def chunk(self,text,no='51',stage='billIntroducedFile',house='Lok Sabha'):
        return {'slot':stage,'part':'narrative','year':'2024','bill_no':no,'text':text,'house':house}
    def block(self,name='ANITA EXAMPLE'):
        return 'STATEMENT OF OBJECTS AND REASONS\nThe Bill seeks to achieve these objects.\nNEW DELHI;\n'+name+'.\nThe 5th April, 2024.'
    def test_pmb_metadata_and_followup(self):
        self.assertIn('Meera Example',self.ask('Which MP introduced Lok Sabha Bill No. 52 of 2024?'))
        self.assertIn('Meera Example',self.ask('Who proposed this bill?'))
    def test_fresh_question_requests_identity(self): self.assertIn('Which bill',self.ask('Who proposed this bill?'))
    def test_respects_wrong_house(self):
        self.ask('Who proposed Lok Sabha Bill No. 52 of 2024?')
        reply=self.ask('Who proposed Rajya Sabha Bill No. 52 of 2024?');self.assertNotIn('Meera Example',reply);self.assertIn('No Rajya Sabha bill',reply)
    def test_only_introduced_and_same_house(self):
        self.brain.chunks={'late':self.chunk(self.block('WRONG LATER'),'51','billPassedInBothHousesFile'), 'wrong':self.chunk(self.block('WRONG HOUSE'),house='Rajya Sabha'), 'right':self.chunk(self.block())}
        s=self.ask('Who proposed Bill No. 51 of 2024?');self.assertIn('Anita Example',s);self.assertIn('signatory',s);self.assertNotIn('Wrong',s)
    def test_place_is_not_a_person(self): self.assertEqual(list(pairs(self.block('NEW DELHI'))),[])
    def test_other_section_is_not_evidence(self): self.assertEqual(list(pairs(self.block().replace('The Bill seeks','FINANCIAL MEMORANDUM\nThe Bill seeks'))),[])
    def test_multiple_signatures_abstain(self):
        self.brain.chunks={'one':self.chunk(self.block()),'two':self.chunk(self.block('BINA EXAMPLE'))}
        s=self.ask('Who proposed Bill No. 51 of 2024?');self.assertIn('No individual',s);self.assertNotIn('Anita',s);self.assertNotIn('Bina',s)
    def test_private_signature_is_not_a_minister(self):
        self.brain.chunks={'ok':self.chunk(self.block(),no='53')};s=self.ask('Who proposed Bill No. 53 of 2024?');self.assertIn('Anita Example',s);self.assertNotIn('minister',s.lower())
    def test_date_before_name(self): self.assertEqual(next(pairs('STATEMENT OF OBJECTS AND REASONS\nNEW DELHI;\nThe 4th March, 2024.\nANITA EXAMPLE.'))[0],'ANITA EXAMPLE')
    def test_counts_from_second_thread(self):
        q='How many bills were introduced in 2024?';self.db.answer(q)
        with ThreadPoolExecutor(max_workers=1) as ex: self.assertIn('3 bills',ex.submit(self.db.answer,q).result()[0])
    def test_nonexistent_house_is_not_substituted(self):
        reply=self.ask('Who proposed Rajya Sabha Bill No. 51 of 2024?');self.assertIn('No Rajya Sabha bill',reply);self.assertNotIn('Example',reply)
    def test_doc_with_text_conversion_is_read(self):
        (self.pdfs/'00002_51_2024_The Archive Study Bill, 2024.doc').write_text('legacy word file',encoding='latin-1')
        (self.pdfs/'00002_51_2024_The Archive Study Bill, 2024.txt').write_text('AS INTRODUCED IN THE LOK SABHA\nBill No. 51 of 2024\n'+self.block(),encoding='utf-8')
        reply=self.ask('Who proposed Bill No. 51 of 2024?');self.assertIn('Anita Example',reply);self.assertIn('existing text conversion',reply)
    def test_wrong_suffix_case_still_matches(self):
        (self.pdfs/'00003_52_2024_The Path Access Bill, 2024.PDF').write_bytes(b'not really a pdf')
        reply=self.ask('Who proposed Bill No. 52 of 2024?');self.assertIn('Meera Example',reply)
    def test_answering_writes_nothing(self):
        before={p:(p.stat().st_mtime_ns,p.stat().st_size) for p in self.root.rglob('*') if p.is_file()}
        for q in ['Who proposed Bill No. 52 of 2024?','When was Bill No. 52 of 2024 passed?','Who proposed this bill?']:
            self.ask(q)
        self.assertEqual(before,{p:(p.stat().st_mtime_ns,p.stat().st_size) for p in self.root.rglob('*') if p.is_file()})
    def test_no_network_during_answers(self):
        import socket
        def refuse(*a,**k): raise AssertionError('the trial must not open a network connection to answer')
        with patch.object(socket.socket,'connect',refuse), patch.object(socket.socket,'connect_ex',refuse):
            self.assertIn('Meera Example',self.ask('Who proposed Bill No. 52 of 2024?'))
    def test_missing_pdf_reader_is_reported_not_guessed(self):
        import attribution
        saved=attribution.EXTRACTOR_MISSING;attribution.EXTRACTOR_MISSING=True
        try:
            reply=self.ask('Who proposed Bill No. 51 of 2024?')
        finally:
            attribution.EXTRACTOR_MISSING=saved
        self.assertIn('cannot read PDFs',reply);self.assertNotIn('Anita',reply)
    def test_local_pdf_without_index(self):
        try: import pymupdf
        except ImportError: self.skipTest('PyMuPDF unavailable in this interpreter')
        with pymupdf.open() as doc:
            pg=doc.new_page();pg.insert_text((64,64),'AS INTRODUCED IN THE LOK SABHA\nBill No. 51 of 2024\n'+self.block());doc.save(self.pdfs/'00001_51_2024_The Archive Study Bill, 2024.pdf')
        reply=self.ask('Who proposed Bill No. 51 of 2024?');self.assertIn('Anita Example',reply);self.assertIn('PDF page 1',reply)

class _UnrelatedBrain:
    """A search index that always returns an unrelated bill's clause, at a high
    score - which is what a vector index does for a bill it does not hold."""
    facts_only = False
    def __init__(self):
        self.last_filters = None
        self.chunks = {"unrelated": {
            "chunk_id": "unrelated", "bill_no": "154", "bill_no_norm": "154",
            "year": "2019", "house": "Lok Sabha",
            "bill_title": "The Road Transport Bill, 2019", "ministry": "ROAD TRANSPORT",
            "status": "Assented", "language": "english", "slot": "billIntroducedFile",
            "part": "operative", "clause_no": "127", "label": "clause 127",
            "source_file": "x.pdf", "source_primary": "sansad",
            "text": "Punishment for non-compliance of directions by Appropriate Commission."}}
    def search(self, question, top_n=3, part=None, filters=None):
        return [{"score": 0.9, "chunk": c} for c in self.chunks.values()][:top_n]
    def chunks_for(self, *a, **k):
        return list(self.chunks.values())
    def year_in_brain(self, year):
        return True


class NoIdentityNoTextAnswerTests(unittest.TestCase):
    def setUp(self):
        self.brain = _UnrelatedBrain()
    def ask(self, q):
        return talk.answer_question(self.brain, q, last_filters=None)

    def test_unmatched_name_is_not_answered_from_another_bill(self):
        reply, hits, used = self.ask("What is the punishment under the POCSO Amendment Bill 2019?")
        self.assertNotIn("non-compliance", reply.lower())
        self.assertNotIn("Road Transport", reply)
        self.assertEqual(hits, [])
        # Two bills share the POCSO name and 2019 is not enough to choose, so
        # asking which one is correct; nothing may be quoted either way.
        self.assertIn(used, ("clarify-bill", "no-such-bill"))
        self.assertNotIn("non-compliance", reply.lower())
    def test_non_bill_question_is_not_answered_from_text(self):
        reply, hits, _ = self.ask("Who is the minister of home affairs?")
        self.assertNotIn("non-compliance", reply.lower())
        self.assertEqual(hits, [])
    def test_number_without_year_is_not_answered_from_text(self):
        reply, hits, used = self.ask("What is the punishment under Bill No. 370?")
        self.assertIn("Which year", reply)
        self.assertEqual(hits, [])
        self.assertEqual(used, "clarify-bill")


class _DocBrain:
    """Index holding ONE bill's text, plus an unrelated bill that always scores
    highest - the situation that produced the wrong-bill POCSO answers."""
    facts_only = False
    def __init__(self, held):
        self.chunks = {}
        for i, (no, yr, house, title, text) in enumerate(held):
            self.chunks[str(i)] = {"chunk_id": str(i), "bill_no": no, "bill_no_norm": no,
                "year": yr, "house": house, "bill_title": title, "ministry": "X",
                "status": "Assented", "language": "english",
                "slot": "billPassedInBothHousesFile", "part": "operative",
                "clause_no": "7", "label": "clause 7", "source_file": "f.pdf",
                "source_primary": "sansad", "text": text}
        self.last_filters = None
    def search(self, q, top_n=3, part=None, filters=None):
        if filters:
            y, no = str(filters[0]), str(filters[1])
            return [{"score": 0.9, "chunk": c} for c in self.chunks.values()
                    if c["year"] == y and c["bill_no"] == no]
        return [{"score": 0.44, "chunk": c} for c in self.chunks.values()][:top_n]
    def chunks_for(self, no, yr, house=None):
        return [c for c in self.chunks.values()
                if c["bill_no"] == str(no) and c["year"] == str(yr)]
    def year_in_brain(self, y):
        return True


class FullModeIdentityTests(unittest.TestCase):
    """With document search ON, an unresolved question must not be answered
    from whichever bill the index happens to rank first."""
    def setUp(self):
        self.brain = _DocBrain([
            ("113", "2023", "Lok Sabha", "THE DIGITAL PERSONAL DATA PROTECTION BILL, 2023",
             "Penalty may extend to two hundred and fifty crore rupees."),
            ("154", "2019", "Lok Sabha", "The Road Transport and Safety Bill, 2019",
             "Penalty for contravention of regulations under section 177."),
        ])
    def ask(self, q):
        return talk.answer_question(self.brain, q, last_filters=None)

    def test_unresolved_bill_never_quoted_from_another_bill(self):
        for q in ["What is the punishment under the POCSO Amendment Bill 2019?",
                  "What is the punishment under the POCSO Amendment Bill?"]:
            reply, hits, used = self.ask(q)
            self.assertNotIn("Road Transport", reply)
            self.assertNotIn("section 177", reply)
            self.assertEqual(hits, [])
            # Whatever the path, nothing from another bill may be quoted.
            self.assertIn(used, ("clarify-bill", "no-such-bill", "scanned-bill"))

    def test_non_bill_question_declines_in_full_mode(self):
        reply, hits, _ = self.ask("Who is the minister of home affairs?")
        self.assertEqual(hits, [])
        self.assertNotIn("section 177", reply)

    def test_list_question_is_answered_from_the_table(self):
        reply, hits, used = self.ask("Which bills were passed in December 2019?")
        self.assertEqual(used, "bill-metadata")
        self.assertIn("8 bills", reply)
        self.assertEqual(hits, [])

    def test_known_bill_still_gets_its_own_text(self):
        reply, hits, used = self.ask("What is the punishment under the DPDP bill?")
        self.assertIn("two hundred and fifty crore", reply)
        self.assertIn("113 of 2023", reply)
        self.assertTrue(hits)

    def test_act_name_reaches_the_bill(self):
        reply, hits, _ = self.ask("What does the DPDP Act say?")
        self.assertIn("113 of 2023", reply)
        self.assertTrue(hits)
        self.assertNotIn("Road Transport", reply)


class PronounFollowUpTests(unittest.TestCase):
    """Seen live: after asking about the CAA bill, "what are all clauses for
    this bill?" answered with two "General Clauses (Amendment)" bills, because
    the word "clauses" (and "general") survived the noise filter and the title
    search runs before the pronoun rule."""
    def setUp(self):
        db = talk.get_facts()
        if db is None:
            self.skipTest("no metadata CSV beside the code")
        self.db = db
        self.ctx = ("2019", "370", "Lok Sabha")

    def resolved(self, question):
        res = self.db.resolve(question, self.ctx)
        return res, [(b["no"], b["year"]) for b in res["bills"]]

    def test_document_words_do_not_hijack_a_follow_up(self):
        for question in ["what are all clauses for this bill?",
                         "what are clauses for this bill",
                         "what does it say",
                         "what are its provisions",
                         "how many clauses does it have"]:
            res, bills = self.resolved(question)
            self.assertEqual(res["kind"], "one", question)
            self.assertEqual(bills, [("370", "2019")], question)

    def test_a_real_title_still_wins(self):
        res, bills = self.resolved("Tell me about the Electricity (Amendment) Bill, 2014")
        self.assertEqual(res["kind"], "one")
        self.assertEqual(bills, [("191", "2014")])

    def test_off_topic_is_still_none(self):
        self.assertEqual(self.resolved("What is the capital of France?")[0]["kind"], "none")
        self.assertEqual(self.resolved("How many bills were introduced in 2026?")[0]["kind"],
                         "none")


class FactsModeCardTests(unittest.TestCase):
    """A text question with no index must print the record it promises - a
    title alone left a 2026 bill with no date at all, seen in a browser."""
    def test_title_question_prints_the_timeline(self):
        db = talk.get_facts()
        if db is None:
            self.skipTest("no metadata CSV beside the code")
        reply, hits, used = talk.answer_question(talk.FactsMode(),
            "Tell me about Lok Sabha Bill No. 144 of 2026", no_llm=True)
        self.assertEqual(used, "facts-mode-needs-index")
        self.assertIn("Timeline", reply)
        self.assertIn("3 August 2026", reply)
        self.assertIn("Pending", reply)
        self.assertIn("Enable document search", reply)
        self.assertNotIn("Document search is off, so I can only give this bill's", reply)


class PlainAnswerShapeTests(unittest.TestCase):
    """The no-LLM composer must not label the short title as a provision, and
    its bill label must read like English. Both were seen live in a browser
    against a real two-PDF brain."""
    def setUp(self):
        def hit(no, yr, part, clause, text):
            return {"score": 0.9, "chunk": {"bill_no": no, "year": yr,
                "house": "Lok Sabha", "bill_title": "THE EXAMPLE BILL, 2019",
                "ministry": "HOME AFFAIRS", "status": "Assented", "language": "english",
                "slot": "billIntroducedFile", "part": part, "clause_no": clause,
                "label": "clause " + clause, "source_file": "f.pdf",
                "source_primary": "sansad", "text": text}}
        self.hits = [
            hit("370", "2019", "narrative", "",
                "STATEMENT OF OBJECTS AND REASONS The Bill amends the principal Act."),
            hit("370", "2019", "operative", "1",
                "1. (1) This Act may be called the Example Act, 2019."),
            hit("370", "2019", "operative", "3",
                "3. A person who contravenes this section shall be punishable with a fine."),
        ]

    def answer(self, q):
        return talk.plain_answer(self.hits, "provision", q, extra=self.hits)

    def test_short_title_is_not_offered_as_a_provision(self):
        for q in ["What does the Example Bill 2019 say?",
                  "What is the punishment under the Example Bill 2019?"]:
            reply = self.answer(q)
            self.assertNotIn("may be called", reply.lower())
            self.assertNotIn("short title", reply.lower())

    def test_real_clause_is_still_quoted(self):
        self.assertIn("punishable with a fine", self.answer(
            "What is the punishment under the Example Bill 2019?"))

    def test_bill_label_reads_as_a_sentence(self):
        reply = self.answer("What does the Example Bill 2019 say?")
        self.assertIn("Bill No. 370 of 2019", reply)
        self.assertNotIn("Bill 370 of 2019,", reply)


class ModeTests(unittest.TestCase):
    """exact vs explain: real only when a model writes the answer."""
    def test_writer_prompt_carries_the_mode(self):
        hits=[{"score":1.0,"chunk":{"bill_no":"113","year":"2023","slot":"x","clause_no":"7","text":"Penalty."}}]
        exact=talk.build_writer_messages("q",hits,{}, "exact")
        explain=talk.build_writer_messages("q",hits,{},"explain")
        self.assertIn("MODE: exact",exact[1]["content"])
        self.assertIn("MODE: explain",explain[1]["content"])
        self.assertNotEqual(exact[1]["content"],explain[1]["content"])
    def test_unknown_mode_falls_back_to_plain_prompt(self):
        hits=[{"score":1.0,"chunk":{"bill_no":"113","year":"2023","slot":"x","clause_no":"7","text":"Penalty."}}]
        out=talk.build_writer_messages("q",hits,{},"")
        self.assertNotIn("MODE:",out[1]["content"])
    def test_local_mode_says_the_toggle_had_no_effect(self):
        brain=_DocBrain([("113","2023","Lok Sabha","THE DIGITAL PERSONAL DATA PROTECTION BILL, 2023",
                          "Penalty may extend to two hundred and fifty crore rupees.")])
        reply,hits,used=talk.answer_question(brain,"What does the DPDP bill say?",
                                             mode="explain",last_filters=None)
        self.assertIn("no model connected",used)
        self.assertTrue(hits)


class _FakeFilters:
    """Stand-in for Brain's last_filters property, to test the threading rule alone."""
    def __init__(self):
        import threading
        self._last_filters=None; self._filters_by_thread={}; self._threading=threading
    @property
    def last_filters(self):
        try: return self._filters_by_thread.get(self._threading.get_ident())
        except AttributeError: return self._last_filters
    @last_filters.setter
    def last_filters(self,value):
        try: self._filters_by_thread[self._threading.get_ident()]=value
        except AttributeError: self._last_filters=value


class ThreadIsolationTests(unittest.TestCase):
    def test_one_session_cannot_inherit_another_sessions_bill(self):
        fake=_FakeFilters();fake.last_filters=('2019','370','Lok Sabha');seen={}
        def other():
            seen['start']=fake.last_filters;fake.last_filters=('2023','II','Rajya Sabha');seen['end']=fake.last_filters
        with ThreadPoolExecutor(max_workers=1) as ex: ex.submit(other).result()
        self.assertIsNone(seen['start'])
        self.assertEqual(seen['end'],('2023','II','Rajya Sabha'))
        self.assertEqual(fake.last_filters,('2019','370','Lok Sabha'))

    def test_real_brain_class_has_the_same_property(self):
        import inspect, step3_build_brain
        source=inspect.getsource(step3_build_brain.Brain.last_filters.fget)
        self.assertIn('get_ident',source)


if __name__=='__main__':unittest.main(verbosity=2)
