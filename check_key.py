"""
check_key.py - small helper to test a Gemini API key without the whole chatbot.

It asks Google for the list of models, trying BOTH ways of passing the key:
  1. in the x-goog-api-key header (what the chatbot now uses)
  2. in the URL (?key=...) (the older way)

It never prints the key itself - only its length and the "AQ."/"AIza" prefix.
Run:  python check_key.py
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def show(label, request):
    """Send one request and print the outcome in a copy-pastable way."""
    print(f"\n--- {label} ---")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
            names = [m.get("name", "?") for m in data.get("models", [])][:5]
            if not names and "choices" in data:   # Groq reply shape
                names = [data["choices"][0]["message"]["content"][:40]]
            print(f"HTTP 200 - THE KEY WORKS. Reply preview: {names}")
            return True
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        print(f"HTTP {error.code}")
        print(body[:500])
    except urllib.error.URLError as error:
        print(f"NETWORK problem (no internet?): {error.reason}")
    return False


def main():
    key = os.environ.get("GEMINI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not key and not groq_key:
        sys.exit("No keys found in this terminal. Set at least one:\n"
                 '  $env:GEMINI_API_KEY="AQ...your key..."\n'
                 '  $env:GROQ_API_KEY="gsk_...your key..."')

    ok1 = ok2 = False
    if not key:
        print("GEMINI_API_KEY not set - skipping the Gemini tests.")
    else:
        prefix = key[:5] if key[:5].startswith(("AQ.", "AIza")) else key[:3]
        print(f"gemini key found: length={len(key)}, starts with '{prefix}..' "
              f"(the full key is never printed)")
        if key != key.strip():
            print("WARNING: key has spaces/quotes around it! Re-set it cleanly.")

        first = urllib.request.Request(
            BASE, headers={"x-goog-api-key": key})  # newer "AQ." keys need a header
        ok1 = show("test 1: key in x-goog-api-key header", first)

        second = urllib.request.Request(f"{BASE}?key={urllib.parse.quote(key)}")
        ok2 = show("test 2: key in URL (?key=...)", second)

    print("\n=== SUMMARY ===")
    if key:
        print("header method works:", ok1)
        print("url method works   :", ok2)
    if key and not ok1 and not ok2:
        print("Neither Gemini method worked - read the HTTP message above;")
        print("paste it to your guide.")

    # The Groq key is the backup provider - test it too when it is set.
    if groq_key:
        print(f"\ngroq key found: length={len(groq_key)}, starts with '{groq_key[:4]}..'")
        groq_headers = {"Content-Type": "application/json",
                        # Groq is behind Cloudflare: a normal User-Agent is
                        # required, else you get HTTP 403 "error code: 1010".
                        "User-Agent": "bills-chatbot/1.0",
                        "Authorization": "Bearer " + groq_key}

        # test 3: can the key CHAT? (uses a model Groq documents as current)
        body = json.dumps({"model": "openai/gpt-oss-20b",
                           "messages": [{"role": "user", "content": "Reply only: OK"}]}).encode()
        groq_req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=body,
            headers=groq_headers)
        groq_ok = show("test 3: Groq chat (backup provider, openai/gpt-oss-20b)", groq_req)
        if groq_ok:
            print("Groq works - the chatbot will use it automatically when Gemini fails.")

        # test 4: WHICH models can this key use? (prints the real list)
        print("\n--- test 4: models available to YOUR Groq key ---")
        models_req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models", headers=groq_headers)
        try:
            with urllib.request.urlopen(models_req, timeout=60) as response:
                data = json.load(response)
            ids = sorted(m.get("id", "?") for m in data.get("data", []))
            print(f"{len(ids)} models available:")
            for model_id in ids:
                print("   ", model_id)
            print("If the chatbot still says a model is missing (404), force one")
            print("of the ids above, for example:")
            print('   $env:GROQ_MODEL="openai/gpt-oss-120b"')
            print('   python step4_talk.py --ask "..."')
        except urllib.error.HTTPError as error:
            print(f"HTTP {error.code}")
            print(error.read().decode('utf-8', 'replace')[:400])
        except urllib.error.URLError as error:
            print(f"NETWORK problem: {error.reason}")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (imported here so the top stays tidy)
    main()
