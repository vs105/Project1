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

from bills_db import BillsDB, find_csv, fmt, norm_title, this_period
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
               r"entail|deal|deals|pass(?:ed|es)?|assent\w*|introduc\w+|"
               r"enact\w+|brought|come\s+into\s+force)\b.*$")
# The same list WITHOUT the trailing ".*$", so every occurrence can be found.
NAME_CLOSER_TAIL = (r"\s+(?:about|do|does|did|prescribe|provide|say|says|cover|"
                    r"covers|replace|repeal|mean|change|amends?|contain|include|"
                    r"entail|deal|deals|pass(?:ed|es)?|assent\w*|introduc\w+|"
                    r"enact\w+|brought|come\s+into\s+force)\b")
# The question words a bill name can never start with, once the leading
# question has been cut off. If a candidate still starts with one of these it
# is a question, not a name.
LEADING_Q = (r"^(?:please\s+)?(?:what|when|why|who|whom|which|how|where)\b"
             r"[\w\s'()-]*?\b(?:is|are|was|were|does|do|did|has|have|had|can|"
             r"could|will|would|shall|should|must)\b\s+")


# A bill name carries on past these words, so a "closer" that is followed by
# one of them was not the end of the question - it was the middle of the name.
# "The Climate Change Bill" ends in "change", which is also a question verb.
NAME_TAIL = re.compile(r"^\s*(?:bill|act|ordinance|amendment|rules?|regulation|"
                       r"\d{4})\b", re.I)


def cut_at_closer(text):
    """
    Cut a question's tail off, leaving the bill's name behind.

    The words in NAME_CLOSER are ordinary English as well as question verbs,
    and bill names use them too. Two tests decide which match is the real end
    of the question: what is left must be a name (at least two words), and
    what follows must not look like the name carrying on. Without the second
    test, "The Climate Change Bill" was cut at "change" and left "climate",
    which matches no bill at all - so the bot fell back to searching the
    whole brain for a question that had named its bill exactly.
    """
    for m in re.finditer(NAME_CLOSER_TAIL, text):
        if NAME_TAIL.match(text[m.end():]):
            continue
        head = re.sub(r"\s*[?.]?\s*$", "", text[:m.start()])
        if len(head.split()) >= 2:
            return head
    return text


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
        had_opener = after_opener != q
        base = after_opener if had_opener else q
        if not had_opener:
            # "When was X passed?" and "What punishment does X prescribe?"
            # put a noun between the question word and the verb, so the strict
            # opener above never matched and the name was silently dropped -
            # the question then searched the whole brain instead of the one
            # bill it had named. Cut the leading question off at the first
            # auxiliary verb instead, which works whatever sits in between.
            # This runs BEFORE the closer, because the closer also matches
            # auxiliary verbs and would otherwise cut the name in half:
            # "what punishment does X prescribe" lost everything after "does".
            loose = re.sub(LEADING_Q, "", base)
            if loose != base:
                base = loose
        after_closer = cut_at_closer(base)
        # A closer is not always there. "tell me about the Finance Bill"
        # simply ends with the name. That is safe for these openers, but not
        # when what follows is another question, so those are kept out.
        had_closer = after_closer != base
        cand = after_closer if had_closer else base
        # A name is only accepted when something actually bounded it: a closer
        # was found, or the strict opener already cleaned the front. Without
        # that, "how many bills were introduced in 2019" would leave
        # "introduced in 2019" behind and be read as a bill name.
        if ((had_closer or had_opener) and cand
                and not re.match(r"(how|which|when|why|what|who)\b", cand)):
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

def title_clash(db, question, plan, bill):
    """
    The one bill the question's TITLE names, when it is not the bill the
    question's NUMBER named. None when the title is missing, ambiguous, or
    agrees with the number - only an unambiguous, DIFFERENT bill is a clash.

    "What is the Digital Personal Data Protection Bill No. 114 of 2023 about?"
    names the DPDP Bill, which is No. 113, and No. 114 of 2023 is the Pharmacy
    (Amendment) Bill. Answering from the number alone silently switched laws.
    """
    # Take the number phrase out and let the ordinary planner name the bill, so
    # exactly the same title rules apply as everywhere else. This is needed
    # because the planner drops the title altogether once it sees a number.
    stripped = re.sub(r"\bbill\s*(?:no\.?|number|num\.?)?\s*\d{1,3}\b"
                      r"(?:\s*(?:of|/|,|\(|-)\s*20[0-2]\d\b)?", " ",
                      question, flags=re.IGNORECASE)
    if stripped.strip() == question.strip():
        return None
    p2 = local_plan(stripped)
    names = list(p2.get("candidate_titles") or [])
    if p2.get("bill_name"):
        names.append(p2["bill_name"])
    for name in names:
        hits = db.find_by_title(name)
        others = [h for h in hits
                  if (str(h["no"]), str(h["year"]))
                  != (str(bill["no"]), str(bill["year"]))]
        if len(hits) == 1 and others:
            return others[0]
    return None


def resolve_bill(db, plan, question, last_filters=None):
    """
    Work out exactly which bill the question is about.

    Returns a dict:
      {"kind": "one",  "bill": {...}}                 lock onto this bill
      {"kind": "many", "bills": [...]}                ask which year
      {"kind": "none", "name": "..."}                 no such bill
      {"kind": "conflict", "bills": [a, b]}           the title and the number
                                                     name two different bills
      {"kind": "skip"}                                no bill was named

    This is where "Triple Talaq" becomes Bill 82 of 2019, and where a name that
    matches nothing is reported as missing instead of being quietly replaced
    with some other bill.
    """
    if plan["bill_number"] and plan["year"]:
        b = db.by_key.get((plan["bill_number"], plan["year"]))
        if b:
            clash = title_clash(db, question, plan, b)
            if clash:
                return {"kind": "conflict", "bills": [b, clash]}
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
            if len(same) > 1:
                # Same name, same year, both a Government bill and a Private
                # Member's bill. The Government bill is the one that can become
                # law, so it is the one the question is about - the same rule
                # that orders every other disambiguation here.
                gov = [b for b in same
                       if str(b.get("type") or "").lower().startswith("gov")]
                if len(gov) == 1:
                    same = gov
            if len(same) == 1:
                return {"kind": "one", "bill": same[0]}
            # Show the bills of the year that was asked about first, so the
            # question "which year" is easy to answer.
            if plan["year"] and same:
                # The user already named a year, so listing bills of OTHER
                # years only makes them answer a question they already
                # answered. When the year is known the list is that year's.
                hits = same
            else:
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


def gov_first(bills):
    """
    Government bills before Private Member's.

    Of the 3,678 bills in this corpus, 932 are Government bills and 2,746 are
    Private Member's. Almost none of the latter ever become law. So when a name
    is shared by both, the Government bill is what the question is about, and a
    Private Member's bill is offered only when no Government bill of that name
    matches at all.
    """
    def is_gov(b):
        return str(b.get("type") or "").lower().startswith("gov")
    return ([b for b in bills if is_gov(b)],
            [b for b in bills if not is_gov(b)])


def which_year(bills, year=None):
    """Ask the user to pick, and list the real options, Government bills first."""
    gov, pmb = gov_first(bills)
    example = (gov or pmb)[0]
    # Ask for the identifier the question is MISSING. A question that already
    # said "of 2023" has answered the year, so replying "Which year?" makes the
    # bot look as though it never read the question - that is exactly what
    # "the Bharatiya Nyaya Sanhita Bill of 2023" got back. Two bills can share
    # a name inside one year, so the bill number is what is left to ask for.
    if year:
        opening = ("I found %d bills called that in %s, so I will not guess "
                   "which one you mean. Which bill number?"
                   % (len(bills), year))
        closing = ('Ask again with the number, for example "Bill No. %s of %s".'
                   % (example["no"], example["year"]))
    else:
        opening = ("I found %d bills with that name, so I will not guess which "
                   "one you mean. Which year?" % len(bills))
        closing = 'Ask again with the year, for example "... of %s".' % (
            example["year"],)
    lines = [opening, ""]
    # The heading must not promise what the list does not deliver: it includes
    # a lapsed bill, and a lapsed Government bill never became law. What is
    # true, and useful, is WHO moved it - the government or a back-bencher -
    # because that is what decides whether it ever had a chance.
    for head, group in (
            ("Government bills - moved by the government, so these are the "
             "ones that can become law:", gov),
            ("Private Member's bills - moved by an individual MP; these "
             "rarely become law:", pmb)):
        if not group:
            continue
        lines.append(head)
        for b in group[:10]:
            lines.append("  - Bill No. %s of %s (%s)"
                         % (b["no"], b["year"], b["status"]))
        if len(group) > 10:
            lines.append("  - and %d more" % (len(group) - 10))
        lines.append("")
    lines.append(closing)
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
    # Set when the bot had to choose between bills and chose one. It is put in
    # front of the answer, because a silent choice is indistinguishable from a
    # fact and the reader can never correct it.
    pick_note = ""

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
        gov, pmb = gov_first(res["bills"])
        # Government bills always come first, and a Private Member's bill is
        # only reached when no Government bill of that name can be answered.
        pool, kind = gov, "Government"
        answerable = [b for b in pool
                      if (str(b.get("no")), str(b.get("year"))) in have]
        if not answerable and pmb:
            pool, kind = pmb, "Private Member"
            answerable = [b for b in pool
                          if (str(b.get("no")), str(b.get("year"))) in have]
        if len(answerable) == 1:
            # Several bills share the name and only one of them has text.
            # Answering from it is the right call - refusing when we can be
            # helpful is its own failure - but it IS a choice between real laws,
            # and the reader must be able to see it and overrule it. So the
            # answer opens by naming what was chosen and what was passed over.
            locked = answerable[0]
            brain.last_filters = (locked["year"], locked["no"])
            others = sorted({b["year"] for b in pool if b is not locked})
            note = ("Several bills share that name, so I am not guessing "
                    "between them - I am telling you which one I used. I hold "
                    "the text of Bill No. %s of %s, a %s bill, so this answer "
                    "comes from that one."
                    % (locked["no"], locked["year"], kind))
            if others:
                note += (" Other %s bills of the same name are from %s, and for "
                         "those I hold the records but not the text."
                         % (kind, ", ".join(others)))
            if kind == "Private Member":
                note += (" No Government bill of that name has text I can read, "
                         "so this is the closest I hold - and a Private Member's "
                         "Bill is a proposal, not law.")
            pick_note = note + " Say the year if you meant a different one."
        elif not answerable:
            # Not one of them can be answered, so the year does not matter.
            return ("Several bills share that name, and I hold the text of none "
                    "of them, so I will not answer about one of them by "
                    "mistake. Try naming the year as well.",
                    [], "no-such-bill")
        else:
            return which_year(res["bills"], plan.get("year")), [], "which-year"

    # A Private Member's Bill is a proposal, not law, and nothing in its text
    # says so. The note is set whenever the bill actually used is a Private
    # Member's one - not only when it was chosen over Government bills -
    # because "the Juvenile Justice Bill of 2025" resolves straight to a PMB
    # and used to say nothing about it at all.
    if (locked and not pick_note
            and str(locked.get("type") or "").lower().startswith("priv")):
        if locked.get("assent") or locked.get("act_no"):
            pick_note = ("Bill No. %s of %s is a Private Member's Bill, but it "
                         "was passed and assented, so this is law."
                         % (locked["no"], locked["year"]))
        else:
            pick_note = ("Bill No. %s of %s is a Private Member's Bill - a "
                         "proposal, not law. Its status is %s and it has not "
                         "been passed, so no Act came from it."
                         % (locked["no"], locked["year"],
                            locked.get("status") or "pending"))

    if res["kind"] == "conflict":
        # Two identities were supplied and they point at different laws. Picking
        # one silently is the same failure as guessing a year.
        numbered, titled = res["bills"]
        return ("Those two do not go together. \"%s\" is Bill No. %s of %s, "
                "but Bill No. %s of %s is \"%s\".\n\n"
                "Which did you mean? Answer with the number and the year "
                "together, for example \"Bill No. %s of %s\"."
                % (titled["title"], titled["no"], titled["year"],
                   numbered["no"], numbered["year"], numbered["title"],
                   titled["no"], titled["year"]), [], "no-such-bill")

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
    # Does this question want the WHOLE bill, or one clause of it? "Tell me
    # about the JJ Bill" wants everything; "what punishment does clause 5
    # prescribe" wants one clause. Reading the whole bill for the second is not
    # thoroughness, it is noise - the answer gets chosen from two hundred
    # clauses instead of the one that matters. The routing above has already
    # sent overview questions to the narrative half, so that is the signal.
    whole_bill = bool(locked and (part == "narrative" or plan["full_text"]))

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
        # the resolved bill has no readable text in the brain. The pick note
        # still belongs here: a Private Member's Bill with no readable text is
        # exactly the one a reader would otherwise take for law.
        return ((pick_note + "\n\n" + scanned_bill_reply(locked)) if pick_note
                else scanned_bill_reply(locked)), [], "scanned-bill"
    if not hits:
        hits = brain.search(search_q, top_n=top_n, filters=filt)
    if not hits:
        if unresolved:
            return no_such_bill(unresolved), [], "no-such-bill"
        if res["kind"] == "skip":
            # Nothing we hold is about the document that was named. This is
            # where a question about an ACT lands: the corpus holds bills, and
            # the Act is not one of them.
            return ("I could not find a bill in my data that this question is "
                    "about, so I will not answer from a different one.\n\n"
                    "I hold 3,678 Lok Sabha BILLS from 2004 to 2026, not Acts. "
                    "If you asked about an Act, ask about the bill that became "
                    "it, or give the bill's number and year - for example "
                    "\"Bill No. 99 of 2014\".", [], "no-such-bill")
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

    # 5a2. With no bill locked, the answer must still come from a bill whose
    #      TITLE relates to the document the question named.
    #
    #      "The Right to Fair Compensation ... Act, 2013" was answered from the
    #      Appropriation (Railways) No. 4 Bill, because that bill's corrigendum
    #      quotes the Act in full - the WORDS matched, the document did not.
    #      Text overlap is not identity, and a semantically similar unrelated
    #      bill must never be allowed to win.
    if not locked and hits:
        qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower())
                  if w not in QUESTION_WORDS}
        if qwords:
            keep = [h for h in hits
                    if qwords & set(re.findall(
                        r"[a-z]{4,}",
                        (h["chunk"].get("bill_title") or "").lower()))]
            if not keep:
                hits = []
            elif len(keep) < len(hits):
                hits = keep
    if not hits and res["kind"] == "skip" and not locked:
        # Nothing we hold is about the document that was named. This is where a
        # question about an ACT lands: the corpus holds bills, and the Act is
        # not one of them.
        return ("I could not find a bill in my data that this question is "
                "about, so I will not answer from a different one.\n\n"
                "I hold 3,678 Lok Sabha BILLS from 2004 to 2026, not Acts. "
                "If you asked about an Act, ask about the bill that became it, "
                "or give the bill's number and year - for example "
                "\"Bill No. 99 of 2014\".", [], "no-such-bill")

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
    digest = None
    if whole_bill:
        # Read the whole bill rather than the three best-matching clauses.
        digest = bill_digest(brain, locked, question, wants)

    if no_llm:
        if digest:
            # Cite the documents the briefing actually read. The answer quotes
            # the gazette version; the sources used to list the top three of a
            # keyword search, one of which was the introduced file - a
            # different document, so no reader could have checked a word.
            shown = [{"score": 1.0, "chunk": c} for c in digest["sources"]]
            return (pick_note + "\n\n" + bill_briefing(digest, locked)
                    if pick_note else bill_briefing(digest, locked),
                    shown, "none")
        # No model, so the answer is composed rather than written. A question
        # about what a bill IS deserves the whole bill, not the half the
        # question happened to be routed to: "what is the Bharatiya Nyaya
        # Sanhita about" reads the preamble, and without the clauses too it can
        # only ever quote that. So when the question is an overview one (the
        # part was switched to narrative), pull the clauses as well.
        more = []
        if locked and part == "narrative":
            try:
                more = brain.search(search_q, top_n=3, part="operative",
                                    filters=filt)
            except Exception:
                more = []
        out = plain_answer(hits, wants, question, more)
        return ((pick_note + "\n\n" + out) if pick_note else out,
                hits + more, "none")
    if mock_llm:
        out = plain_answer(hits, wants, question)
        return ((pick_note + "\n\n" + out) if pick_note else out, hits, "mock")
    writer_hits = hits
    if digest:
        # Same whole bill, handed to the model instead of the top three
        # clauses. Capped because a hundred-clause bill is far more tokens than
        # one request should carry; the digest already holds the structure, so
        # what is sent here is the text the model will actually quote.
        pool = digest["chunks"]
        keep = [c for c, _ in digest["asked"]]
        for c in pool:
            if len(keep) >= 40:
                break
            if not any(c is k for k in keep):
                keep.append(c)
        writer_hits = [{"score": 1.0, "chunk": c} for c in keep]
    try:
        text, used = groq_chat(build_writer_messages(question, writer_hits,
                                                     plan, mode),
                               api_key, llm_models or LLM_MODELS)
        return ((pick_note + "\n\n" + text) if pick_note else text,
                writer_hits, used)
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
    # The year is optional: "Notwithstanding anything contained in the Code of
    # Criminal Procedure, any child who has completed sixteen years..." is the
    # same opener with no citation at all, and leaving it in made a clause
    # about children read like a clause about procedure.
    r"Notwithstanding anything contained in [^,]{0,90},\s*(?:\d{4}\s*,?)?",
]

# What a clause can start with once the boilerplate is gone: a leftover year, a
# stray dash, a clause letter. None of it is an answer.
LEADING_JUNK = r"^[\d\s,;:.\-]*\(?[a-z]\)?[\s,;:.\-]*"

# Section furniture. In the bill these words sit on a line of their own; PDF
# extraction glues them onto whatever sentence follows, so they have to come
# off a SENTENCE and not only off the start of a chunk. A reader shown
# "STATEMENT OF OBJECTS AND REASONS The bill was enacted to..." is being shown
# the bill's filing system instead of its reason.
# Every word is separated by \s+ rather than a single space because PDF
# extraction leaves double spaces in headings as often as single ones - the
# Bharatiya Nyaya Sanhita's Statement of Objects is spelled "STATEMENT  OF
# OBJECTS  AND  REASONS" and a plain-space pattern never finds it.
HEAD_WORDS = (
    r"STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS",
    r"NOTES\s+ON\s+CLAUSES",
    r"FINANCIAL\s+MEMORANDUM",
    r"MEMORANDUM\s+REGARDING\s+DELEGATED\s+LEGISLATION",
    r"ARRANGEMENT\s+OF\s+CLAUSES",
    r"THE\s+GAZETTE\s+OF\s+INDIA(?:\s*EXTRAORDINARY)?"
    r"(?:\s*\[?\s*PART\s+[IVXLC]+[^\]]{0,30}\]?)?",
    r"PUBLISHED\s+BY\s+AUTHORITY",
    r"AS\s+PASSED\s+BY\s+THE\s+(?:HOUSES\s+OF\s+PARLIAMENT|LOK\s+SABHA"
    r"|RAJYA\s+SABHA)",
)

SOO = r"STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS"
NOTES = r"NOTES\s+ON\s+CLAUSES"


# Furniture printed after a long title in the published PDFs: a stage marker in
# brackets, and the gazette's own file number. The Digital Personal Data
# Protection Bill was being described as "...incidental thereto. (As passed by
# the Houses of Parliament) MGIPMRND-306RS-16-08-2023."
TITLE_TAIL = re.compile(
    r"[.;]\s*(?:\(\s*AS\s+(?:INTRODUCED|PASSED)\b[^)]{0,70}\)"
    r"|\b[A-Z]{3,}[A-Z0-9]*-\d+[A-Z0-9\-]*)", re.IGNORECASE)


def cut_at_heading(s):
    """
    Truncate at the first section heading, stage marker or gazette file number.

    A long title is quoted with a 420-character window so that a title broken
    across a page still comes out whole, and that window runs straight on into
    the next section: the Bharatiya Nyaya Sanhita was being described as
    "...incidental thereto. STATEMENT OF OBJECTS AND REASONS In the year 1834,
    the first Indian Law Commission...". A heading follows a full stop, so the
    cut is made there and the full stop is kept.
    """
    m = re.search(r"[.;]\s*(?:%s)" % "|".join(HEAD_WORDS), s, re.IGNORECASE)
    t = TITLE_TAIL.search(s)
    if m and (not t or m.start() <= t.start()):
        return s[:m.start() + 1]
    return s[:t.start() + 1] if t else s


def strip_headings(s):
    """Take section furniture off the front of a sentence, however many
    headings the extractor glued together."""
    for _ in range(4):
        new = re.sub(r"^\s*(?:%s)\s*[:\-\u2014\u2013]*\s*" % "|".join(HEAD_WORDS),
                     " ", s, flags=re.IGNORECASE)
        if new == s:
            break
        s = new
    return s

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
    s = strip_headings(s)
    s = re.sub(LEADING_JUNK, "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    # "shall be punished" reads as an answer, "shall be liable to" does not.
    s = s.replace("shall also be liable to fine", "is also liable to a fine")
    s = s.replace("shall be", "is").replace("shall also", "is also")
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    return s


def reason_sentence(text, qwords=()):
    """
    The sentence in a Statement of Objects that says why the bill was brought.

    The Statement is a short numbered argument: what the law was, what was
    wrong with it, what the Government decided. The sentence carrying the
    decision is the answer, and it is a long one - so candidates are filtered
    for length FIRST and scored after. Scoring first picked the Sanhita's
    closing line, "The Bill seeks to achieve the above objectives.", because it
    contains the words "the Bill seeks"; that line was then thrown away for
    being under ninety characters and the answer fell through to a different
    section of the bill altogether.
    """
    sents = [x for x in re.split(r"(?<=[.;])\s+", text) if len(x) > 25]
    pool = [x for x in sents if len(x) >= 90] or sents
    best, top = None, -1
    for sent in pool:
        low = sent.lower()
        score = 0
        for w in ("expedient", "necessary", "proposed to enact", "it is proposed",
                  "in order to", "with a view to", "the government",
                  "considered", "object of the bill"):
            if w in low:
                score += 3
        # A paragraph that only restates the bill's own name says nothing.
        if re.search(r"seeks to achieve the above", low):
            score -= 6
        score += 5 * sum(1 for w in qwords if w in low)
        if score > top:
            best, top = sent, score
    return plain_sentence(best or (pool[0] if pool else text))


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


# A bill is published as up to eight SEPARATE documents: the draft as
# introduced, the version passed in each House, the version passed in both,
# the gazette, the synopsis, the committee report. They are not the same text.
# A clause can be amended, renumbered or deleted between the draft and the
# gazette, so quoting the draft and calling it "what the bill prescribes" is
# quoting a provision that may never have become law.
#
# So every piece of text this assistant quotes says which document it came
# from, and where a clause exists in more than one, the enacted version wins.
SLOT_LABEL = {
    "billGazettedFile": "the gazette version",
    "billPassedInBothHousesFile": "as passed by both Houses",
    "billPassedInLSFile": "as passed by the Lok Sabha",
    "billPassedInRSFile": "as passed by the Rajya Sabha",
    "billIntroducedFile": "as introduced",
    "billSynopsisFile": "the synopsis",
    "reportFile": "the committee report",
    "errataFile": "the errata sheet",
}

# Lower is more authoritative. The gazette is the law as enacted; the introduced
# file is a draft that was later changed. A clause that exists only in the
# draft is still shown - labelled as the draft - because refusing to answer is
# worse than answering with the version we hold.
# How each document is named when the answer says where it read the bill from.
# The article is carried here rather than bolted on outside, because
# "from the the gazette version" and "from the as introduced" both read as
# typos in the answer.
SPINE_NAME = {
    "billGazettedFile": "the gazette version",
    "billPassedInBothHousesFile": "the version passed by both Houses",
    "billPassedInLSFile": "the Lok Sabha-passed version",
    "billPassedInRSFile": "the Rajya Sabha-passed version",
    "billIntroducedFile": "the introduced version",
    "billSynopsisFile": "the synopsis",
    "reportFile": "the committee report",
    "errataFile": "the errata sheet",
}

SLOT_RANK = {
    "billGazettedFile": 0,
    "billPassedInBothHousesFile": 1,
    "billPassedInLSFile": 2,
    "billPassedInRSFile": 3,
    "billIntroducedFile": 4,
    "billSynopsisFile": 5,
    "reportFile": 6,
    "errataFile": 7,
}


def source_of(chunk):
    """Name the document a piece of text came from, for the reader."""
    slot = chunk.get("slot") or ""
    return SLOT_LABEL.get(slot, slot or "the bill text")


def bill_digest(brain, bill, question, wants):
    """
    Read EVERY chunk this brain holds for one bill, and distil it.

    Top-K search answers "which clause matches this question". This answers
    "tell me about this bill", where the honest move is to read the whole
    thing. The digest is the bill's own structure: what it says it is, why it
    was brought, what it covers, and what it prescribes on the thing that was
    actually asked.

    It is built once and used twice - formatted for the reader when there is
    no model, and handed to the model when there is one - so both paths answer
    from the same whole bill rather than the same three clauses.
    """
    mine = brain.chunks_for(bill["no"], bill["year"])
    if not mine:
        return None
    qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower())
              if w not in QUESTION_WORDS}
    # The bill's own name must not count as a topical match. "What is the
    # Juvenile Justice Bill about" shares juvenile and justice with half the
    # clauses in it, so without this the answer's clause list is chosen by the
    # title rather than by what was asked. The bill is already identified, so
    # its name tells us nothing about which clause matters.
    qwords -= set(norm_title(bill["title"]).split())

    # 1. What the bill is: its own long title. Every Indian bill opens with
    #    "A BILL to ..." and that sentence is the bill describing itself.
    purpose, purpose_src, purpose_c = "", "", None
    # The long title is quoted from the most authoritative document we hold.
    # The versions differ, and the enacted one is the one that counts - the
    # Juvenile Justice Bill was describing itself from the Lok Sabha version
    # while the gazette sat in the same brain.
    titled = sorted((c for c in mine
                     if re.search(r"\bA\s+BILL\s+to\b", c["text"], re.IGNORECASE)),
                    key=lambda c: SLOT_RANK.get(c.get("slot"), 9))
    for c in titled:
        m = re.search(r"\bA\s+BILL\s+to\b.{0,420}", c["text"],
                      re.IGNORECASE | re.DOTALL)
        if m:
            purpose = plain_sentence(cut_at_heading(m.group(0)))
            if len(purpose) > 420:
                purpose = purpose[:420].rsplit(" ", 1)[0]
            purpose_src, purpose_c = source_of(c), c
            break

    # 2. Why it exists: the Statement of Objects and Reasons, in the bill's own
    #    words. That section is the bill explaining itself, so it is looked for
    #    BY NAME before anything else is considered.
    #
    #    Without that, an overview question was answered out of NOTES ON
    #    CLAUSES, which is the reason for ONE clause: "why was the Bharatiya
    #    Nyaya Sanhita brought?" came back "Liability of abettor for an effect
    #    caused by the act abetted different from that intended...". Notes on
    #    Clauses is still read, but only when the Statement is not held.
    why, why_src, why_c = "", "", None
    for head in (SOO, NOTES):
        # Same preference: the enacted document's own words first.
        pool = sorted((c for c in mine if c.get("part") == "narrative"
                       and re.search(head, c["text"], re.IGNORECASE)),
                      key=lambda c: SLOT_RANK.get(c.get("slot"), 9))
        for c in pool:
            body = re.sub(r"^.*?" + head + r"\s*", "", c["text"],
                          flags=re.IGNORECASE | re.DOTALL)
            cand = reason_sentence(body, qwords)
            # "Clause 63 of the Bill seeks to..." explains ONE clause. On a
            # question about the whole bill that is not the reason the bill was
            # brought, so it is refused rather than mislabelled - the Digital
            # Personal Data Protection Bill was being introduced with the
            # duties of a Data Principal as its reason for existing.
            if re.match(r"\s*(?:this\s+)?clause\b", cand, re.IGNORECASE):
                continue
            if len(cand) > 90 and cand[:70] not in purpose[:220]:
                why, why_src, why_c = cand, source_of(c), c
                break
        if why:
            break

    # 3. What it covers: the chapters and the clauses inside each. This is the
    #    skeleton of the whole law and it costs a dozen lines, so a reader can
    #    see the scope of a hundred-clause bill without reading a hundred
    #    clauses.
    #    Read the skeleton from ONE version of the bill, never merged across
    #    versions. A bill is published as several documents and the same clause
    #    is numbered differently in the draft and in the Act, so taking the
    #    lowest and highest number seen for a chapter across every version we
    #    hold produced a skeleton that ran backwards and overlapped:
    #    "CHAPTER II: clauses 3-10" followed by "CHAPTER III: clauses 5-9", and
    #    for the Finance (No. 2) Bill 2024 a "CHAPTER VI: clauses 1-168" that
    #    was simply the whole bill. One document is internally consistent; four
    #    merged are not. The version we hold most completely is the one to
    #    read, and a tie goes to the more authoritative document.
    op = [c for c in mine
          if c.get("part") == "operative"
          and re.match(r"\d+", c.get("clause_no") or "")]
    per_slot = {}
    for c in op:
        per_slot.setdefault(SLOT_RANK.get(c.get("slot"), 9), []).append(c)
    spine_slot = (min(per_slot, key=lambda x: (-len(per_slot[x]), x))
                  if per_slot else None)
    spine = per_slot.get(spine_slot, []) if spine_slot is not None else []
    # Which document the skeleton was read from. A bill's chapters are numbered
    # differently between the draft and the Act, so a skeleton quoted without
    # its version is a claim the reader cannot check.
    spine_src = SPINE_NAME.get(spine[0].get("slot"), "the introduced version") \
        if spine else ""
    terms, seen = [], {}
    for c in spine:
        num = int(re.match(r"\d+", c["clause_no"]).group(0))
        # "CHAPTER  IV" with two spaces is the PDF's spacing, not the chapter's
        # name, and it reads like a typo in the answer.
        t = re.sub(r"\s+", " ", c.get("term") or "").strip()
        if t not in seen:
            seen[t] = [num, num, t]
            terms.append(seen[t])
        else:
            seen[t][0] = min(seen[t][0], num)
            seen[t][1] = max(seen[t][1], num)
    # A skeleton is only worth printing if the chapter labels hold together:
    # each chapter's clauses must begin after the previous chapter's clauses
    # ended. When they do not, the labels are wrong for this document - the
    # arrangement-of-clauses list at the front of a bill is set in the same
    # typeface as its chapters, so its headings get attached to clause 1, and
    # the Juvenile Justice Bill came back "CHAPTER II: clauses 3-10" followed
    # by "CHAPTER III: clauses 5-9". A skeleton that runs backwards is worse
    # than no skeleton, so the bill is then described by its span instead.
    clean = []
    for lo, hi, t in terms:
        if clean and lo <= clean[-1][1]:
            clean = None
            break
        clean.append([lo, hi, t])
    terms = clean or []

    # 4. What it prescribes on the thing that was asked. One version of each
    #    clause, the most authoritative document we hold, and only clauses that
    #    carry a word the question used - so the whole bill is read but the
    #    answer is still about what was asked.
    best = {}
    for c in mine:
        if c.get("part") != "operative" or not c.get("clause_no"):
            continue
        rank = SLOT_RANK.get(c.get("slot"), 9)
        cur = best.get(c["clause_no"])
        if cur is None or rank < cur[0]:
            best[c["clause_no"]] = (rank, c)
    asked = []
    for rank, c in sorted(best.values(), key=lambda rc: rc[0]):
        low = c["text"].lower()
        # Clause 1 of every Indian Act is the short title, and it repeats the
        # bill's whole name - so on a "what is X about" question it out-matches
        # every real clause. It is the same trap as the title page.
        if re.search(r"\bmay be called\b|\bshort title\b", low):
            continue
        hits_words = sum(1 for w in qwords if w in low)
        grounds = any(w in low for w in GROUNDING.get(wants, ()))
        if not (grounds or hits_words >= 1):
            continue
        sent = answer_sentence(c["text"], wants, qwords)
        if len(sent) > 60:
            asked.append((c, sent))
        if len(asked) == 3:
            break

    # How much of the bill this is. The count is of DISTINCT clauses, not of
    # chunks: a clause published as the draft, the Lok Sabha version and the
    # gazette is one clause held three times, and counting chunks reported
    # "I hold 245 clauses" for a bill with 112.
    held = {c["clause_no"] for c in op}
    nums = {int(re.match(r"\d+", n).group(0)) for n in held}

    # Every clause the skeleton does not account for is named rather than left
    # as a silent gap. Three chapters of the Juvenile Justice Act were coming
    # back one clause short - Chapter III as "clauses 5-9" when the Act puts
    # sections 4 to 9 there, Chapter VIII as 57-73 when it is 56 to 73 - and
    # nothing on screen said so, so the map read as complete when it was not.
    covered = set()
    for lo, hi, _ in terms:
        covered.update(range(lo, hi + 1))
    gaps = (sorted(n for n in nums if n not in covered) if terms else [])

    # The documents this briefing actually read, so the source list can cite
    # them. Citing the top three of a keyword search instead showed a chunk
    # from the introduced file while the answer quoted the gazette, and no
    # reader could have checked a word of it.
    srcs = []
    for c in [purpose_c, why_c] + [x[0] for x in asked]:
        if c is not None and not any(c is u for u in srcs):
            srcs.append(c)
    if not srcs:
        srcs = (op or mine)[:3]

    return {"purpose": purpose, "purpose_src": purpose_src,
            "why": why, "why_src": why_src, "terms": terms, "asked": asked,
            "n_clauses": len(held), "top_clause": max(nums) if nums else 0,
            "slots": sorted({c.get("slot") for c in mine if c.get("slot")},
                            key=lambda x: SLOT_RANK.get(x, 9)),
            "sources": srcs, "spine_src": spine_src, "gaps": gaps,
            "chunks": mine}


def bill_briefing(d, bill):
    """
    Format a whole-bill digest as an answer a person can read.

    It says what the bill is, why it exists, what it covers, and what it
    prescribes on the question - and it says which document each of those came
    from, because a bill is published several times and the versions differ.
    """
    out = [bill["title"], ""]
    line = "Bill %s of %s" % (bill["no"], bill["year"])
    if bill.get("ministry"):
        line += "  -  " + bill["ministry"]
    out.append(line)

    dates = [("%s %s" % (lbl, fmt(bill.get(k))))
             for lbl, k in (("introduced", "intro"), ("passed LS", "ls"),
                            ("passed RS", "rs"), ("assented", "assent"))
             if bill.get(k)]
    if bill.get("act_no"):
        dates.append("Act %s of %s" % (bill["act_no"], bill.get("act_year") or ""))
    if dates:
        out.append("  ".join(dates))

    if d["purpose"]:
        out.append("")
        out.append("What it is: %s (%s)" % (d["purpose"], d["purpose_src"]))
    if d["why"]:
        out.append("")
        out.append("Why it was brought: %s (%s)" % (d["why"], d["why_src"]))
    if d["terms"]:
        out.append("")
        out.append("What it covers%s:" % ((" (from %s)" % d["spine_src"])
                                           if d.get("spine_src") else ""))
        for lo, hi, t in d["terms"]:
            name = t if t else "the main provisions"
            rng = ("clauses %d-%d" % (lo, hi)) if lo != hi else ("clause %d" % lo)
            out.append("  - %s: %s" % (name, rng))
        if d["gaps"]:
            few = d["gaps"][:8]
            out.append("  (held but under no chapter heading in this text: "
                       + ("clause %s" if len(few) == 1 else "clauses %s")
                       % ", ".join(str(n) for n in few)
                       + ("" if len(d["gaps"]) <= len(few)
                          else " and %d more" % (len(d["gaps"]) - len(few)))
                       + ")")
    elif d["top_clause"]:
        # The chapter labels for this document do not hold together, so the
        # bill is described by its span rather than by chapters we cannot
        # vouch for.
        out.append("")
        out.append("What it covers:")
        out.append("  - the main provisions: clauses 1-%d" % d["top_clause"])
    if d["asked"]:
        out.append("")
        out.append("On what you asked:")
        for c, sent in d["asked"]:
            out.append("  - Clause %s: %s (%s)"
                       % (c["clause_no"], sent, source_of(c)))

    # Coverage, said plainly. A bill published in eight documents may reach us
    # as one of them, and the reader deserves to know which.
    out.append("")
    cov = "I hold %d clauses of this bill" % d["n_clauses"]
    if d["top_clause"] and d["n_clauses"] < d["top_clause"]:
        cov += (" (the text runs to clause %d, so some is missing)"
                % d["top_clause"])
    cov += ", from " + ", ".join(source_of({"slot": x}) for x in d["slots"]) + "."
    out.append(cov)
    return "\n".join(out)


def plain_answer(hits, wants, question, extra=()):
    """
    Compose an answer out of the bill's own words. No model, no network, and
    nothing invented.

    With no language model there is no way to WRITE a sentence, only to choose
    and order the ones the bill already contains. So the answer is built in the
    shape a person expects - what the bill is, why it exists, what it
    prescribes - and each part is taken from the part of the bill that actually
    says it. The caller passes the operative clauses in `extra`, because the
    preamble answers "what is this" and the clauses answer "what does it do",
    and one question usually deserves both.

    It is still the bill's words. It is no longer one clause with a label
    stapled to the front of it.
    """
    pool = list(hits)
    for h in extra:
        if not any(h["chunk"] is x["chunk"] for x in pool):
            pool.append(h)
    if not pool:
        return REFUSAL

    c0 = pool[0]["chunk"]
    bill = "Bill %s of %s" % (c0.get("bill_no"), c0.get("year"))
    ministry = (c0.get("ministry") or "").strip()
    title = (c0.get("bill_title") or "").strip()
    qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower())
              if w not in QUESTION_WORDS}

    # 1. What the bill is: its own long title. Every Indian bill opens with
    #    "A BILL to ..." and that single sentence is the bill describing
    #    itself, which is exactly what "what is this about" asks for.
    purpose, purpose_src = "", ""
    for h in pool:
        m = re.search(r"\bA\s+BILL\s+to\b.{0,420}", h["chunk"]["text"],
                      re.IGNORECASE | re.DOTALL)
        if m:
            purpose = plain_sentence(cut_at_heading(m.group(0)))
            if len(purpose) > 420:
                purpose = purpose[:420].rsplit(" ", 1)[0]
            purpose_src = source_of(h["chunk"])
            break

    # 2. Why it exists: the Statement of Objects and Reasons. Skipped when it
    #    is the very chunk the long title came from, so the same words are
    #    never printed twice under two headings.
    why, why_src = "", ""
    # The same rule as the whole-bill digest uses: the bill's own Statement of
    # Objects by name first, Notes on Clauses only if that is not held, and
    # never a note that explains one clause. A narrow question gets the same
    # "why" an overview question gets, or none.
    for head in (SOO, NOTES):
        for h in pool:
            c = h["chunk"]
            if c.get("part") != "narrative":
                continue
            if not re.search(head, c["text"], re.IGNORECASE):
                continue
            body = re.sub(r"^.*?" + head + r"\s*", "", c["text"],
                          flags=re.IGNORECASE | re.DOTALL)
            cand = reason_sentence(body, qwords)
            if re.match(r"\s*(?:this\s+)?clause\b", cand, re.IGNORECASE):
                continue
            if len(cand) > 90 and cand[:70] not in purpose[:220]:
                why, why_src = cand, source_of(c)
                break
        if why:
            break

    # 3. What it prescribes: one version of each clause, the most authoritative
    #    one we hold, and only clauses that carry a word the question actually
    #    used - so a clause is never quoted merely for being clause 2. A clause
    #    that shares nothing is kept as a spare rather than thrown away, because
    #    a vague question still deserves an answer.
    best = {}
    for h in pool:
        c = h["chunk"]
        if c.get("part") != "operative" or not c.get("clause_no"):
            continue
        rank = SLOT_RANK.get(c.get("slot"), 9)
        cur = best.get(c["clause_no"])
        if cur is None or rank < cur[0]:
            best[c["clause_no"]] = (rank, h)
    good, spare = [], []
    for rank, h in sorted(best.values(), key=lambda rh: rh[0]):
        c = h["chunk"]
        sent = answer_sentence(c["text"], wants, qwords)
        if len(sent) <= 60 or any(sent[:60] in g for g in good):
            continue
        low = c["text"].lower()
        strong = (any(w in low for w in GROUNDING.get(wants, ()))
                  or any(w in low for w in qwords))
        line = "Clause %s %s (%s)" % (c["clause_no"], sent, source_of(c))
        (good if strong else spare).append(line)
        if len(good) == 2:
            break
    prescribes = good or spare[:1]

    body = []
    if purpose:
        body.append("What it is: %s (%s)" % (purpose, purpose_src))
    if why:
        body.append("Why it was brought: %s (%s)" % (why, why_src))
    if prescribes:
        body.append("What it prescribes:\n"
                    + "\n".join("  - " + x for x in prescribes))

    head = title if title else ""
    tail = "%s%s" % (bill, ", " + ministry if ministry else "")
    if body:
        return (head + "\n" if head else "") + tail + "\n\n" + "\n\n".join(body)

    # Nothing composed: a bill whose text is all operative, or a question the
    # composer found no shape for. Fall back to the best single sentence
    # rather than say nothing at all.
    c = pool[0]["chunk"]
    clause = c.get("clause_no") or ""
    sent = answer_sentence(c["text"], wants, qwords)
    frame = FRAMES.get(wants, FRAMES["provisions"]) % (clause or "n/a")
    if not clause:
        frame = "The bill's own background note says this."
    return "%s %s (%s.)" % (frame, sent, bill)


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
