"""
STEP 4 - THE TALK LAYER

Turns a question into an answer from real bill data.

What changed, and why it matters:

THE OLD WAY matched the question against hand written word lists. So "when was
the Finance Bill passed" worked, "tell me about the Finance Bill" did not, and
"I need the passing date of the Finance Bill" did not either. The same question
asked a different way reached a different layer and got a wrong answer. No list
of words can cover how people actually ask things.

THE NEW WAY lets Groq read the question and say what it means. Groq returns a
PLAN, never a fact. The plan is then checked against the real 3,678 bill titles
and executed by ordinary code. The rule that keeps this honest:

    GROQ MAY PROPOSE. ONLY REAL DATA MAY CONFIRM.

So Groq can guess that "Triple Talaq" means the Muslim Women (Protection of
Rights on Marriage) Bill, and the code then checks whether that bill really
exists. If it does not, the bot says so instead of answering from some other
bill.

Three things the bot can now say that it could not say before:
  1. the bill exists, here is the answer
  2. the bill exists but its file is a scanned image, so it cannot be read yet
  3. no bill of that name exists in 2004 to 2026

Run it like this:
    python step4_talk.py --ask "what punishment does the Muslim Women Bill 2019 prescribe"
    python step4_talk.py --interactive
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from bills_db import BillsDB, find_csv, norm_title, this_period
from step3_build_brain import Brain, MODEL_NAME, TOP_N, extract_filters, tokenize

TIMEOUT = 60
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq is the one provider this project uses. The gpt-oss models come first
# because Groq retired the llama-3.3 and llama-3.1 ids for free keys. The old
# names stay last because they still work on paid plans.
LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b",
              "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

REFUSAL = ("I could not find this in the official bill text I hold, so I would "
           "rather not guess. Try naming the bill and the year, or ask what the "
           "bill does, what it provides, or when it was passed.")

# Small talk. Kept as a tiny local list on purpose: a greeting must be instant
# and must never cost an API call.
SMALLTALK = [
    (r"^\s*(hi|hey|hello|good\s*(morning|afternoon|evening)|namaste)\b",
     "Hello. Ask me anything about Lok Sabha bills from 2004 to 2026. You can "
     "name a bill, ask what it does, ask when it was passed, or ask how many "
     "bills there were in a year."),
    (r"\b(thanks|thank you|thx)\b",
     "You are welcome. Ask another question whenever you want."),
    (r"\b(bye|goodbye|see you)\b",
     "Goodbye. Everything here comes from official Sansad bill documents."),
    (r"\b(help|what can you do|how do you use this)\b",
     "Ask me about bills. For example: what does the Bharatiya Nyaya Sanhita "
     "replace, when was the Citizenship Amendment Bill passed, how many bills "
     "were introduced in 2019, or tell me about the Finance Bill 2024. If a "
     "bill is not in the 2004 to 2026 corpus I will say so."),
]


# ----------------------------------------------------------------------------
# GROQ - one helper for both jobs (planning and writing)
# ----------------------------------------------------------------------------

class AuthError(Exception):
    """The key was rejected. Nothing to retry, so say it plainly."""


def groq_chat(messages, api_key, models, json_mode=False):
    """
    Send messages to Groq and return the text reply.

    json_mode=True asks Groq to answer with a JSON object only, which is what
    the question planner needs.
    """
    failures = []
    for model in models:
        body = {"model": model, "messages": messages, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            GROQ_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "User-Agent": "bills-chatbot/1.0",
                     "Authorization": "Bearer " + api_key})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code in (401, 403) or "invalid_api_key" in msg:
                raise AuthError("Groq rejected the API key (HTTP %d).\n"
                                "Groq's own reply: %s" % (e.code, msg[:300]))
            failures.append("%s: HTTP %d %s"
                            % (model, e.code, msg[:120].replace("\n", " ")))
            continue
        except urllib.error.URLError:
            raise
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            failures.append(model + ": empty reply")
            continue
        if text:
            return text, "groq:" + model
    raise RuntimeError("Groq did not answer with any of these models:\n  "
                       + "\n  ".join(failures))


# ----------------------------------------------------------------------------
# THE PLANNER - what is the user actually asking for
# ----------------------------------------------------------------------------

PLAN_SCHEMA = """Reply with ONE json object and nothing else. Use these keys:
{"intent": "metadata"|"content"|"conversation"|"off_topic",
 "bill_name": "the bill the user named, copied as they wrote it, or null",
 "candidate_titles": ["the official title of that bill, as printed in Indian law"],
 "bill_number": 113 or null,
 "year": 2020 or null,
 "wants": "date"|"status"|"ministry"|"count"|"list"|"penalty"|"provisions"|
          "definition"|"communities"|"why"|"explanation",
 "aggregate": "count"|"list"|null,
 "full_text": false,
 "refers_previous": false}

Rules for filling it in:
- intent "metadata" is for questions about dates, status, ministry, how many,
  or which bills. intent "content" is for questions about what a bill says or
  does. intent "conversation" is greetings and thanks. intent "off_topic" is
  anything that is not about bills at all.
- candidate_titles is your best guess at the OFFICIAL title. Indian bill titles
  are uppercase and end with a year, like "THE BHARATIYA NYAYA SANHITA, 2023".
  If the user uses a nickname like "Triple Talaq", "CAA" or "farm laws", put
  the real official title you believe it refers to. If you do not know, leave
  it empty. Never invent a title you are not confident about.
- bill_number is only set when the user gave a number.
- year is only set when the user gave a year or said "this year".
- wants "penalty" for punishment, "why" for the reason a bill was brought,
  "provisions" for what it changes, "definition" for what a term means,
  "communities" for which groups are covered.
- refers_previous is true when the question says "it", "that", "the same" or
  otherwise points back at the previous question.
- full_text is true only when the user asks for all details, full text or
  everything.
"""

PLAN_SYSTEM = ("You are the question parser for a chatbot about Lok Sabha "
               "bills of India, 2004 to 2026. You read a user question and "
               "describe what it is asking. You never answer it and you never "
               "state a fact about any bill.\n\n" + PLAN_SCHEMA)


def groq_plan(question, history, api_key, models):
    """Ask Groq what the question means. Returns a plan dict, or None."""
    msgs = [{"role": "system", "content": PLAN_SYSTEM}]
    for q, a in (history or [])[-3:]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": (a or "")[:300]})
    msgs.append({"role": "user", "content": question})
    try:
        text, used = groq_chat(msgs, api_key, models, json_mode=True)
    except AuthError:
        raise
    except Exception:
        return None
    return validate_plan(text)


def validate_plan(text):
    """
    Turn Groq's reply into a plan we trust enough to act on.

    Groq's output is never trusted blindly. Every field is checked and given a
    safe default, so a strange reply degrades to "content question about
    nothing in particular" instead of crashing or inventing a filter.
    """
    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            return None
        try:
            raw = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None

    plan = {
        "intent": "content",
        "bill_name": None,
        "candidate_titles": [],
        "bill_number": None,
        "year": None,
        "wants": "explanation",
        "aggregate": None,
        "full_text": False,
        "refers_previous": False,
    }
    intent = str(raw.get("intent") or "").strip().lower()
    if intent in ("metadata", "content", "conversation", "off_topic"):
        plan["intent"] = intent
    name = raw.get("bill_name")
    if isinstance(name, str) and len(name.strip()) >= 3:
        plan["bill_name"] = name.strip()[:200]
    titles = raw.get("candidate_titles")
    if isinstance(titles, list):
        plan["candidate_titles"] = [str(t).strip()[:200] for t in titles
                                    if isinstance(t, str) and len(t.strip()) >= 5][:4]
    num = raw.get("bill_number")
    if isinstance(num, int) and 1 <= num <= 9999:
        plan["bill_number"] = str(num)
    yr = raw.get("year")
    if isinstance(yr, int) and 1900 <= yr <= 2100:
        plan["year"] = str(yr)
    wants = str(raw.get("wants") or "").strip().lower()
    if wants in ("date", "status", "ministry", "count", "list", "penalty",
                 "provisions", "definition", "communities", "why", "explanation"):
        plan["wants"] = wants
    agg = str(raw.get("aggregate") or "").strip().lower()
    if agg in ("count", "list"):
        plan["aggregate"] = agg
    plan["full_text"] = bool(raw.get("full_text"))
    plan["refers_previous"] = bool(raw.get("refers_previous"))
    return plan


# How a question wraps a bill name. "what is the Bharatiya Nyaya Sanhita 2023
# about" -> the name is everything between the opener and the closer.
NAME_OPENER = (r"^(?:please\s+)?(?:tell\s+me\s+(?:about|what|more)|"
               r"what\s+is|what\s+are|what\s+does|what\s+did|"
               r"describe|explain|give\s+me|summari[sz]e|show\s+me)\s+"
               r"(?:the\s+)?")
NAME_CLOSER = (r"\s+(?:about|do|does|did|prescribe|provide|say|says|cover|"
               r"covers|replace|repeal|mean|change|amends?|contain|include|"
               r"entail|deal|deals)\b.*$")


def local_plan(question, last_filters=None):
    """
    A plan built without Groq, for when there is no key or no internet.

    It is much weaker than the model planner and that is the whole point of
    the rewrite: this is the shape of the old behaviour. It exists so the bot
    still works offline, not because it is good enough.
    """
    q = (question or "").lower()
    plan = {"intent": "content", "bill_name": None, "candidate_titles": [],
            "bill_number": None, "year": None, "wants": "explanation",
            "aggregate": None, "full_text": False, "refers_previous": False}
    if re.search(r"\b(hi|hello|hey|thanks|thank you|bye)\b", q):
        plan["intent"] = "conversation"
        return plan
    # The year is read FIRST, and a bill number is only taken when it is
    # clearly not a year. "the Muslim Women Bill 2019" used to be read as bill
    # number 2019, which then matched no bill at all.
    yrs = [int(y) for y in re.findall(r"\b(20[0-2]\d)\b", q)]
    if yrs:
        plan["year"] = str(yrs[0])
    m = re.search(r"bill\s*(?:no\.?|number)\s*(\d{1,4})", q)
    if not m:
        m = re.search(r"\bbill\s+(\d{1,3})\b", q)
    if m:
        n = m.group(1)
        if not (len(n) == 4 and 1900 <= int(n) <= 2100):
            plan["bill_number"] = n
    # The bill NAME. Without this the offline planner can only ever resolve a
    # bill by number, so "what is the Bharatiya Nyaya Sanhita about" locked onto
    # nothing and searched the whole brain.
    #
    # The name is what sits between a question opener and a question closer, and
    # BOTH must have matched. That is what keeps "what punishment does the
    # Muslim Women Bill prescribe" from becoming its own bill name: it has a
    # closer but no opener, so there is nothing to take the name out of.
    if not plan["bill_number"]:
        after_opener = re.sub(NAME_OPENER, "", q)
        if after_opener != q:
            after_closer = re.sub(NAME_CLOSER + r"\s*[?.]?\s*$", "",
                                  after_opener)
            # A closer is not always there. "tell me about the Finance Bill"
            # simply ends with the name. That is safe for these openers, but not
            # when what follows is another question, so those are kept out.
            had_closer = after_closer != after_opener
            cand = after_closer if had_closer else after_opener
            if had_closer or not re.match(r"(how|which|when|why|what|who)\b",
                                          cand):
                name = re.sub(r"\s+", " ", cand).strip(" ,.:?-")
                if (len(name) >= 10
                        and not re.search(r"\b(it|its|that|this|these|those)\b",
                                          name)
                        and not re.match(r"(definition|meaning|punishment|clause|"
                                         r"section|difference|reason|purpose)\b",
                                         name)):
                    plan["bill_name"] = name[:200]
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q):
        plan["intent"] = "metadata"
        plan["wants"] = "count"
        plan["aggregate"] = "count"
    elif (re.search(r"\b(list|show|which|name)\b", q)
          and not re.search(r"\bwhich (communities|groups|people|persons|"
                            r"classes|sectors)\b", q)):
        plan["intent"] = "metadata"
        plan["wants"] = "list"
        plan["aggregate"] = "list"
    elif re.search(r"\bwhen\b|\bdate\b|\bpassed\b|\bassent", q):
        plan["intent"] = "metadata"
        plan["wants"] = "date"
    elif re.search(r"\bpunish|\bpenalty|\bfine\b|\bimprison", q):
        plan["wants"] = "penalty"
    elif re.search(r"\bwhy\b|\breason\b|\bbackground\b", q):
        plan["wants"] = "why"
    elif re.search(r"\bdefine\b|\bdefinition\b|\bmeans\b|\bwhat does .* mean\b", q):
        plan["wants"] = "definition"
    elif re.search(r"\bwhich (communities|groups|people)\b", q):
        plan["wants"] = "communities"
    else:
        plan["wants"] = "provisions"
    if re.search(r"\ball details\b|\bfull text\b|\beverything\b", q):
        plan["full_text"] = True
    # A follow-up question names no bill of its own. It either points back with
    # a word like "it", or it is simply short. Either way, when the previous
    # question was about one bill, this one almost certainly is too. Without
    # this, "what punishment does it prescribe" searched the whole brain and
    # came back with a clause from an unrelated bill.
    names_a_bill = (plan["bill_number"] or plan["year"]
                    or re.search(r"\b(bill|act)\b", q))
    if last_filters and any(last_filters) and not names_a_bill:
        if (re.search(r"\b(it|its|that|the same|this|these|those)\b", q)
                or len(q.split()) <= 8):
            plan["refers_previous"] = True
    return plan


# ----------------------------------------------------------------------------
# THE RESOLVER - turn a name into a real bill, or say there is none
# ----------------------------------------------------------------------------

def resolve_bill(db, plan, question, last_filters=None):
    """
    Work out exactly which bill the question is about.

    Returns a dict:
      {"kind": "one",  "bill": {...}}                 lock onto this bill
      {"kind": "many", "bills": [...]}                ask which year
      {"kind": "none", "name": "..."}                 no such bill
      {"kind": "skip"}                                no bill was named

    This is where "Triple Talaq" becomes Bill 82 of 2019, and where a name that
    matches nothing is reported as missing instead of being quietly replaced
    with some other bill.
    """
    if plan["bill_number"] and plan["year"]:
        b = db.by_key.get((plan["bill_number"], plan["year"]))
        if b:
            return {"kind": "one", "bill": b}
        return {"kind": "none",
                "name": "Bill No. %s of %s" % (plan["bill_number"], plan["year"])}

    # Names to try: the model's guess at the official title first, then whatever
    # the user actually wrote. Both are checked against the real titles, so a
    # confident guess by the model cannot become a fact.
    names = list(plan["candidate_titles"])
    if plan["bill_name"]:
        names.append(plan["bill_name"])
    for name in names:
        # The year is NOT used to filter here, only to choose between matches.
        #
        # A bill is filed under the year it was introduced, but its title often
        # carries a different year, the year of the Act it amends or the year it
        # was drafted. "The Juvenile Justice (Care and Protection of Children)
        # Bill, 2015" is Bill 99 of 2014. Filtering on the year in the title
        # threw that bill away and the answer became "no such bill" for a bill
        # that is in the data. 68 of 3,678 bills were lost that way.
        #
        # An exact title match is stronger evidence than a year someone typed,
        # so the title is looked up on its own first and the year only breaks a
        # tie. Three more bills were being locked onto the wrong one because of
        # this, including "THE WILD LIFE (PROTECTION) AMENDMENT BILL, 2022",
        # which is Bill 159 of 2021.
        hits = db.find_by_title(name)
        if len(hits) == 1:
            return {"kind": "one", "bill": hits[0]}
        if len(hits) > 1:
            same = [b for b in hits if b["year"] == plan["year"]]
            if len(same) != 1 and plan["year"]:
                # The year a user types is the one printed in the TITLE, and a
                # bill is filed under the year it was INTRODUCED, which is often
                # the year before. "The Juvenile Justice (Care and Protection of
                # Children) Bill, 2015" is Bill 99 of 2014. Four bills share
                # that title and only one carries 2015 in its name, so matching
                # on the introduction year alone found none of them and the bot
                # asked "which year" about a question that had named one.
                # 671 titles are shared by more than one bill, so this is not a
                # corner case.
                same = [b for b in hits
                        if re.search(r"\b%s\b" % re.escape(str(plan["year"])),
                                     b["title"] or "")]
            if len(same) == 1:
                return {"kind": "one", "bill": same[0]}
            # Show the bills of the year that was asked about first, so the
            # question "which year" is easy to answer.
            hits = same + [b for b in hits if b not in same]
            return {"kind": "many", "bills": hits[:12]}
    if names:
        return {"kind": "none", "name": plan["bill_name"] or names[0]}

    # Nothing named. A follow-up like "what punishment does it prescribe" can
    # still inherit the bill from the previous question.
    if plan["refers_previous"] and last_filters and any(last_filters):
        old_year, old_no = last_filters
        if old_no and old_year:
            b = db.by_key.get((str(old_no), str(old_year)))
            if b:
                return {"kind": "one", "bill": b}
        if old_year:
            same = [b for b in db.bills if b["year"] == str(old_year)]
            if len(same) == 1:
                return {"kind": "one", "bill": same[0]}
    return {"kind": "skip"}


def no_such_bill(name):
    """The third outcome, said plainly."""
    return ("I could not find a bill called \"" + str(name)[:120] + "\" in the "
            "Lok Sabha bills from 2004 to 2026 that I hold. So I will not "
            "answer about it, because answering about a different bill would "
            "be worse than saying I do not have it.\n\n"
            "Check the exact title, or add the year. Some bills are known by a "
            "nickname, so try the official name, for example \"The Muslim "
            "Women (Protection of Rights on Marriage) Bill, 2019\" rather than "
            "\"the Triple Talaq bill\".")


def which_year(bills):
    """Ask the user to pick, and list the real options."""
    lines = ["I found %d bills with that name, so I will not guess which one "
             "you mean. Which year?" % len(bills), ""]
    for b in bills:
        lines.append("- Bill No. %s of %s (%s, %s)"
                     % (b["no"], b["year"], b["type"], b["status"]))
    lines.append("")
    lines.append("Ask again with the year, for example \"... of 2019\".")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# THE GROUNDING CHECK - is the text we found really about the question
# ----------------------------------------------------------------------------

# What the answer must contain for each kind of question. If the retrieved
# clause has none of these words, it is probably not answering the question,
# even though it scored well, and we say so instead of quoting it.
GROUNDING = {
    "penalty": ("imprison", "fine", "punish", "penalty", "term which",
                "years", "bailable", "cognizable"),
    "definition": ("means", "refers to", "definition", "shall have the same "
                    "meaning", "unless the context"),
    "communities": ("hindu", "sikh", "buddhist", "jain", "parsi", "christian",
                    "community", "communities", "class of persons"),
    "why": ("proposed", "apprehensions", "it has been decided", "in order to",
            "therefore", "the bill seeks", "object", "reason"),
}

# Which half of a bill to read for each kind of question. A punishment question
# must not be answered from the minister's background note, and a "why was this
# brought" question must not be answered from a clause.
PART_FOR = {
    "penalty": "operative",
    "provisions": "operative",
    "definition": "operative",
    "communities": "operative",
    "why": "narrative",
    # "Tell me about X" and "what does X do" are answered from the bill's own
    # Objects and Reasons and its "A BILL to..." preamble, not from a clause in
    # the middle of it. Without this, "what is the Bharatiya Nyaya Sanhita
    # about" was handed clause 128, which defines assault, and the writer had
    # to say it could not answer. Measured: the narrative half returns the
    # preamble for all three bills tested, the mixed search does not.
    "explanation": "narrative",
}


# Words that appear in the question but say nothing about the subject. Used
# only when checking whether a clause defines the term being asked about.
QUESTION_WORDS = {
    "what", "which", "does", "define", "defined", "definition", "means",
    "mean", "under", "with", "from", "that", "this", "there", "about",
    "would", "could", "should", "give", "tell", "explain", "according",
    "please", "bill", "act", "clause", "section", "india", "indian",
}


def grounding_ok(hits, wants, question=""):
    """True when at least one retrieved clause really covers what was asked."""
    need = GROUNDING.get(wants)
    if not need:
        return True
    for h in hits:
        low = h["chunk"]["text"].lower()
        if any(w in low for w in need):
            return True
    if wants == "definition":
        # A clause can define a term without using the word "means". Clause 3
        # of the Muslim Women Bill 2019 defines talaq just by describing it,
        # with none of the words in GROUNDING, so refusing there was wrong.
        # If the question names a term and the clause contains that term,
        # treat that as a definition.
        terms = {w for w in re.findall(r"[a-z]{5,}", question.lower())
                 if w not in QUESTION_WORDS}
        for h in hits:
            low = h["chunk"]["text"].lower()
            if any(t in low for t in terms):
                return True
    return False


# ----------------------------------------------------------------------------
# THE WRITER - Groq explains what the code found, and nothing else
# ----------------------------------------------------------------------------

WRITER_SYSTEM = """You explain Indian Lok Sabha bills to a student. You are given
extracts from official bill documents and a question.

Rules you must follow:
1. Use ONLY the extracts. If the extracts do not contain the answer, say so in
   one sentence and stop. Never fill a gap from memory.
2. Answer only what was asked. If the question is about one thing, answer that
   one thing. Do not add background that was not asked for.
3. Do not quote long passages. Say what the bill does in your own words, and
   name the clause number when the extract shows one.
4. Never invent a date, a number, a ministry or a clause number.
5. Write 2 to 4 sentences unless the question asks for all details.
"""


def build_writer_messages(question, hits, plan, mode):
    parts = [WRITER_SYSTEM, "", "Extracts from official bill documents:"]
    for n, h in enumerate(hits, 1):
        c = h["chunk"]
        label = ("Bill %s of %s, %s, clause %s"
                 % (c.get("bill_no"), c.get("year"), c.get("slot"),
                    c.get("clause_no") or "n/a"))
        parts.append("[%d] %s\n%s" % (n, label, c["text"][:2500]))
    parts.append("")
    if plan.get("full_text"):
        parts.append("The user asked for full details, so cover every point the "
                     "extracts support, in order.")
    parts.append("Question: " + question)
    parts.append("Answer:")
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": "\n".join(parts)}]


# ----------------------------------------------------------------------------
# THE FACTS SHORTCUT - dates and counts, exact, no LLM
# ----------------------------------------------------------------------------

_DB = None


def get_facts():
    """The facts layer, loaded once. Works with no key and no internet."""
    global _DB
    if _DB is None:
        path = find_csv()
        if not path:
            return None
        _DB = BillsDB(path)
    return _DB


def facts_reply(db, question, plan):
    """
    Answer a metadata question exactly, from the bill table.

    The plan tells us this is a metadata question and what it wants, so the
    old keyword triggers are no longer what decides. The table is still read
    with the same code that passed 300 out of 300 bill-number checks.
    """
    reply, _key = db.answer(question, None)
    return reply


# ----------------------------------------------------------------------------
# THE ONE ENTRY POINT
# ----------------------------------------------------------------------------

def answer_question(brain, question, mode="exact", api_key="", llm_models=None,
                    top_n=TOP_N, history=None, last_filters=None,
                    no_llm=False, mock_llm=False):
    """
    Returns (reply, hits, used).

    used is one of: conversation, sansad-metadata, groq:<model>, planner-failed,
    no-such-bill, which-year, refusal, none.
    """
    question = (question or "").strip()
    if not question:
        return "Ask me something about a bill.", [], "conversation"

    # 1. Small talk. Instant, no API call, never touches the brain.
    for pat, reply in SMALLTALK:
        if re.search(pat, question, re.IGNORECASE):
            return reply, [], "conversation"

    db = get_facts()

    # 2. Understand the question. Groq if we have a key, local rules if not.
    llm_error = None
    if no_llm or mock_llm or not api_key:
        plan = local_plan(question, last_filters)
    else:
        try:
            plan = groq_plan(question, history, api_key,
                             llm_models or LLM_MODELS)
        except (AuthError, SystemExit):
            # A wrong, expired or empty key must not stop the assistant from
            # answering. Fall back to the local rules and say which happened.
            plan = None
            llm_error = "the Groq key was rejected, so this answer came from the local rules"
        if plan is None:
            plan = local_plan(question, last_filters)

    if plan["intent"] == "off_topic":
        return ("I only answer questions about Lok Sabha bills from 2004 to "
                "2026. Ask me what a bill does, when it was passed, or how "
                "many bills there were in a year.", [], "off_topic"

                )
    if plan["intent"] == "conversation":
        for pat, reply in SMALLTALK:
            if re.search(pat, question, re.IGNORECASE):
                return reply, [], "conversation"
        return ("Ask me about bills from 2004 to 2026.", [], "conversation"

                )

    # 3. Work out which bill, honestly.
    res = resolve_bill(db, plan, question, last_filters) if db else {"kind": "skip"}
    unresolved = res.get("name") if res["kind"] == "none" else None
    locked = res.get("bill")
    if locked:
        brain.last_filters = (locked["year"], locked["no"])
    if res["kind"] == "many":
        # Asking "which year" is only useful when the answer would change what
        # we can say. Four bills are called "the Citizenship Amendment Bill",
        # in 2014, 2015, 2016 and 2019, and the brain holds the text of exactly
        # one of them, so the others are metadata only and there is nothing to
        # answer from.
        #
        # The same trick separates the two Bharatiya Nyaya Sanhitas of 2023,
        # Bill 173 Passed and Bill 121 Withdrawn, which no year can tell apart
        # because both are 2023. The brain holds the text of one.
        have = brain_bills(brain)
        answerable = [b for b in res["bills"]
                      if (str(b.get("no")), str(b.get("year"))) in have]
        if len(answerable) == 1:
            locked = answerable[0]
            brain.last_filters = (locked["year"], locked["no"])
        elif not answerable:
            # Not one of them can be answered, so the year does not matter.
            return ("Several bills share that name, and I hold the text of none "
                    "of them, so I will not answer about one of them by "
                    "mistake. Try naming the year as well.",
                    [], "no-such-bill")
        else:
            return which_year(res["bills"]), [], "which-year"

    # 4. Metadata questions go to the bill table. Exact, free, no LLM.
    if plan["intent"] == "metadata" and not plan["full_text"]:
        reply = facts_reply(db, question, plan) if db else None
        if reply:
            return reply, [], "sansad-metadata"

    # 5. Content questions go to the bill text, restricted to the right half.
    wants = plan["wants"]
    part = PART_FOR.get(wants)
    # An overview question wants the bill's own Objects and Reasons, not a
    # clause from the middle of it. The planner calls these "provisions", which
    # would read the operative half, so the shape of the question decides
    # instead. A question that names a clause or a section number is asking
    # about that clause, so it keeps the operative half.
    if (locked and part == "operative"
            and re.search(r"\b(tell me about|overview|summari)\b"
                          r"|\bwhat (is|are)\b.*\babout\b"
                          r"|\bwhat does\b.*\bdo\b\s*\??\s*$",
                          question, re.IGNORECASE)
            and not re.search(r"\b(clause|section)\s+\d+", question,
                              re.IGNORECASE)):
        part = "narrative"
    if part == "narrative" and not locked:
        # "Tell me about X" wants the bill's own Objects and Reasons, but that
        # is only a safe preference once we know which bill we are in. Search
        # the whole brain for the narrative half and it lands in whichever bill
        # happens to have the longest preamble: asked about the Constitution
        # 106th Amendment Bill, it came back quoting the Finance (No. 2) Bill.
        # With no bill locked, read every part and let 5a identify the bill.
        part = None
    search_q = question
    filt = None
    if locked:
        # Search inside the one bill we resolved. This is a database filter,
        # so it costs nothing and removes the whole wrong-bill class of error.
        #
        # The filter is PASSED IN, not appended to the question. Appending
        # "bill 99 of 2014" used to be how the filter reached the search, but
        # the search reads the FIRST year in the string, and the first year is
        # the one in the bill's title. So a question about the Juvenile Justice
        # Bill, 2015 resolved correctly to Bill 99 of 2014 and was then searched
        # as Bill 99 of 2015 - a Culture bill - and answered with its financial
        # memorandum. The resolver was right and the search overruled it.
        search_q = "%s bill %s of %s" % (question, locked["no"], locked["year"])
        filt = (locked["year"], locked["no"])
    hits = brain.search(search_q, top_n=top_n, part=part, filters=filt)
    if locked and not hits:
        # the resolved bill has no readable text in the brain
        return scanned_bill_reply(locked), [], "scanned-bill"
    if not hits:
        hits = brain.search(search_q, top_n=top_n, filters=filt)
    if not hits:
        if unresolved:
            return no_such_bill(unresolved), [], "no-such-bill"
        return REFUSAL, [], "refusal"

    # 5a. If the answer spans several bills but the question clearly names one
    # of them, keep only that bill.
    #
    # With a Groq key the planner says which bill was meant. Without one, this
    # is the substitute, and it is pure matching against the real titles, no
    # guessing: "the Muslim Women Protection of Rights on Marriage Bill 2019"
    # shares six words with Bill 82's title and none with Bill 370's, even
    # though both are bills of 2019.
    if hits:
        by_bill = {}
        for h in hits:
            key = (h["chunk"].get("bill_no"), h["chunk"].get("year"))
            by_bill.setdefault(key, []).append(h)
        if len(by_bill) > 1:
            qwords = set(re.findall(r"[a-z]{4,}", question.lower()))
            best, best_score = None, 0
            for key, hs in by_bill.items():
                twords = set(re.findall(
                    r"[a-z]{4,}", hs[0]["chunk"].get("bill_title", "").lower()))
                score = len(qwords & twords)
                if score > best_score:
                    best, best_score = key, score
            if best and best_score >= 3:
                hits = [h for h in hits
                        if (h["chunk"].get("bill_no"),
                            h["chunk"].get("year")) == best]

    # 5b. Remember which bill we just answered about.
    #
    # A follow-up like "what punishment does it prescribe" names no bill, so it
    # has to inherit one. When the question did not name a bill but every
    # extract we found came from a single bill, that bill IS the subject of the
    # conversation, so record it. Without this, "it" resolved to "any bill of
    # 2019" and the answer came back quoting an unrelated bill.
    if not locked and hits:
        bills_in = {(h["chunk"].get("bill_no"), h["chunk"].get("year"))
                    for h in hits}
        if len(bills_in) == 1:
            no, yr = next(iter(bills_in))
            if no and yr:
                brain.last_filters = (yr, no)

    # 6. Check the text really covers what was asked.
    if not grounding_ok(hits, wants, question):
        other = "narrative" if part == "operative" else "operative"
        alt = brain.search(search_q, top_n=top_n, part=other, filters=filt)
        if grounding_ok(alt, wants):
            hits = alt
        else:
            return ("I found Bill %s of %s in my data, but the text I hold does "
                    "not cover this, so I will not guess."
                    % (locked["no"], locked["year"]) if locked else REFUSAL,
                    [], "refusal")

    # 7. Explain it.
    #
    # With a working key, the model writes the answer. Without one, the
    # explainer below writes it from the clauses already retrieved. Either way
    # the user gets sentences, never a dump of quoted text.
    if no_llm:
        return plain_answer(hits, wants, question), hits, "none"
    if mock_llm:
        return plain_answer(hits, wants, question), hits, "mock"
    try:
        text, used = groq_chat(build_writer_messages(question, hits, plan, mode),
                               api_key, llm_models or LLM_MODELS)
        return text, hits, used
    except AuthError as e:
        # A missing or rejected key must not mean a blank screen or a dump of
        # quotes. Say which of the two it is, because the fix is different.
        why = ("no Groq key is set, so this answer was written from the "
               "official text rather than explained by the model"
               if not api_key else
               "the Groq key was rejected, so this answer was written from "
               "the official text rather than explained by the model")
        return (plain_answer(hits, wants, question) + "\n\n(" + why + ".)",
                hits, "planner-failed")
    except Exception as e:
        return (plain_answer(hits, wants, question) +
                "\n\n(The model could not be reached: %s. The official text "
                "is still exact.)" % type(e).__name__,
                hits, "planner-failed")


def brain_bills(brain):
    """(bill_no, year) for every bill the brain actually holds text for."""
    try:
        vals = (brain.chunks.values() if hasattr(brain.chunks, "values")
                else brain.chunks)
        return {(str(c.get("bill_no")), str(c.get("year"))) for c in vals}
    except Exception:
        return set()


def scanned_bill_reply(bill):
    """The second outcome: the bill is real but its text is a scan."""
    return ("Bill No. %s of %s, \"%s\", is in my data and I can tell you its "
            "dates and status, but the text I hold for it is a scanned image "
            "with no text layer, so I cannot quote its clauses yet.\n\n"
            "Ask me when it was introduced or passed and I can answer that "
            "from the official metadata."
            % (bill["no"], bill["year"], bill["title"][:90]))


# ----------------------------------------------------------------------------
# THE OFFLINE EXPLAINER - answer in sentences when there is no API key
# ----------------------------------------------------------------------------

# Fixed legal phrasing that carries no answer. Taking it out is safe because
# it is boilerplate, not substance, and it is what makes a clause read like a
# clause instead of like an answer.
BOILERPLATE = [
    r"referred to in section \d+[A-Z]*",
    r"referred to in clause \([.\w]+\)",
    r"under this Act",
    r"for the purposes of this Act",
    r"Without prejudice to the generality of the provisions of this Act",
    # The year belongs to the Act being cited, so take it with the citation.
    r"Notwithstanding anything contained in [^,]{0,90},\s*\d{4}\s*,?",
]

# What a clause can start with once the boilerplate is gone: a leftover year, a
# stray dash, a clause letter. None of it is an answer.
LEADING_JUNK = r"^[\d\s,;:.\-]*\(?[a-z]\)?[\s,;:.\-]*"

# How to open an answer for each kind of question. The sentence that follows
# comes from the bill itself, so the frame is the only thing written here.
FRAMES = {
    "penalty": "The bill sets out the punishment in clause %s.",
    "definition": "Clause %s defines it.",
    "communities": "Clause %s sets out who is covered.",
    "why": "The bill's own note on this is in clause %s.",
    "provisions": "Clause %s of the bill deals with this.",
}


def plain_sentence(text):
    """Trim a clause down to the sentence that answers, in readable words."""
    s = text
    for pat in BOILERPLATE:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*\d+[A-Z]?\.\s*", "", s)          # leading clause number
    s = re.sub(LEADING_JUNK, "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    # "shall be punished" reads as an answer, "shall be liable to" does not.
    s = s.replace("shall also be liable to fine", "is also liable to a fine")
    s = s.replace("shall be", "is").replace("shall also", "is also")
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    return s


def answer_sentence(text, wants, qwords):
    """Pick the one sentence in a clause that best answers the question."""
    sents = [x for x in re.split(r"(?<=[.;])\s+", text) if len(x) > 25]
    if not sents:
        return plain_sentence(text)
    best, top = None, -1
    for sent in sents:
        low = sent.lower()
        score = sum(2 for w in qwords if w in low)
        score += 4 * sum(1 for w in GROUNDING.get(wants, ()) if w in low)
        # A sentence that is only a cross-reference is never the answer.
        if len(low) < 40:
            score -= 5
        if score > top:
            best, top = sent, score
    return plain_sentence(best or sents[0])


def plain_answer(hits, wants, question):
    """
    Write an answer in plain sentences using only the clauses already
    retrieved. No model, no network, no invented facts. This is what the
    assistant says when no API key is available, and it is the difference
    between answering and dumping.
    """
    h = hits[0]
    c = h["chunk"]
    clause = c.get("clause_no") or ""
    qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower())
              if w not in QUESTION_WORDS}
    sent = answer_sentence(c["text"], wants, qwords)
    frame = FRAMES.get(wants, FRAMES["provisions"]) % (clause or "n/a")
    bill = "Bill %s of %s" % (c.get("bill_no"), c.get("year"))
    if not clause:
        frame = "The bill's own background note says this."
    out = "%s %s" % (frame, sent)
    if len(hits) > 1:
        nxt = hits[1]["chunk"]
        if (nxt.get("clause_no") and nxt["clause_no"] != clause
                and nxt.get("part") == "operative"):
            extra = answer_sentence(nxt["text"], wants, qwords)
            # A marginal note or a cross-reference is not worth a second
            # sentence, and a wrong second sentence is worse than none.
            if (len(extra) > 60 and "etc." not in extra
                    and extra[:40] not in sent
                    and any(w in nxt["text"].lower()
                            for w in GROUNDING.get(wants, ()))):
                out += " Clause %s adds: %s" % (nxt["clause_no"], extra)
    return out + " (%s.)" % bill


def chunks_only(hits):
    """Chunks-only mode: quote the official text as it stands, labelled."""
    out = []
    for n, h in enumerate(hits, 1):
        c = h["chunk"]
        head = ("[%d] Bill %s of %s, %s%s"
                % (n, c.get("bill_no"), c.get("year"), c.get("slot"),
                   ", clause " + c["clause_no"] if c.get("clause_no") else ""))
        out.append(head + "\n" + c["text"][:1400])
    return ("No language model is connected, so here is the official text I "
            "hold, quoted as it stands:\n\n" + "\n\n".join(out))


def format_sources(hits):
    if not hits:
        return "  (no document was needed for that answer)"
    out = []
    for n, h in enumerate(hits, 1):
        c = h["chunk"]
        out.append("  [%d] Bill %s of %s | %s | %s | %s | score %s"
                   % (n, c.get("bill_no"), c.get("year"), c.get("ministry"),
                      c.get("slot"), c.get("part"), h["score"]))
    return "\n".join(out)


# ----------------------------------------------------------------------------
# COMMAND LINE
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Step 4: talk to your bills brain")
    ap.add_argument("--ask", default="", help="answer one question and exit")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--mode", default="exact", choices=["exact", "explain"])
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--out", default="./brain")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="MUST match the model the brain was built with")
    ap.add_argument("--llm-model", default="", help="force one model")
    ap.add_argument("--api-key", default="", help="Groq key, else GROQ_API_KEY")
    ap.add_argument("--no-llm", action="store_true",
                    help="chunks only, no key needed")
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--rerank", action="store_true",
                    help="re-score candidates with a cross-encoder (better "
                         "clause choice, about 1.3s more per question)")
    ap.add_argument("--show-chunks", action="store_true")
    a = ap.parse_args()

    key = (a.api_key or os.environ.get("GROQ_API_KEY", "")).strip()
    if not a.no_llm and not a.mock_llm and not key:
        sys.exit("\nERROR: no Groq key found.\n"
                 "Groq is free and needs no card: https://console.groq.com\n"
                 "Set it in PowerShell:  $env:GROQ_API_KEY=\"gsk_...\"\n"
                 "Or skip the model:  python step4_talk.py --ask \"...\" --no-llm\n")
    models = [a.llm_model] if a.llm_model else LLM_MODELS

    print("Loading brain from " + a.out + " ...")
    brain = Brain(a.out, a.model, rerank=a.rerank)
    db = get_facts()
    if db:
        print("Facts layer: %d bills (dates, counts and lists need no key)." % len(db.bills))
    else:
        print("Facts layer: not loaded (bills_metadata.csv not found).")

    history, last_filters = [], None

    def handle(question):
        nonlocal last_filters
        reply, hits, used = answer_question(brain, question, a.mode, key, models,
                                            a.top_n, history, last_filters,
                                            a.no_llm, a.mock_llm)
        last_filters = getattr(brain, "last_filters", (None, None))
        history.append((question, reply))
        print("\nQ: " + question + "\n" + "-" * 62)
        print(reply)
        print("\nSources (via " + used + "):\n" + format_sources(hits))
        if a.show_chunks and hits:
            print("\nRaw extracts:\n" + chunks_only(hits))

    if a.ask:
        handle(a.ask)
        return
    if not a.interactive:
        ap.print_help()
        return
    print("\nChat mode. Ask about bills from 2004 to 2026. Ctrl+C to leave.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if q:
            try:
                handle(q)
            except AuthError as e:
                print("\n" + str(e))


if __name__ == "__main__":
    main()
