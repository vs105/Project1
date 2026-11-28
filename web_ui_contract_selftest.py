#!/usr/bin/env python3
"""Dependency-light static tests for the Flask/Streamlit integration contract.

Run: python web_ui_contract_selftest.py
This does not start a web server or load the Chroma brain.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class WebUIContractTests(unittest.TestCase):
    def test_answer_question_signature_and_keyword_calls(self):
        backend = ast.parse(read("step4_talk.py"))
        fn = next(
            node for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "answer_question"
        )
        expected = {
            "mode", "api_key", "llm_models", "top_n", "history",
            "last_filters", "no_llm", "mock_llm",
        }
        self.assertTrue(expected.issubset({arg.arg for arg in fn.args.args}))

        for filename in ("app.py", "server.py"):
            tree = ast.parse(read(filename))
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "answer_question"
            ]
            self.assertEqual(len(calls), 1, filename)
            self.assertEqual(len(calls[0].args), 2, filename)
            names = {keyword.arg for keyword in calls[0].keywords}
            self.assertTrue(expected.issubset(names), (filename, names))

    def test_streamlit_no_stale_api(self):
        app = read("app.py")
        self.assertNotIn("try_facts", app)
        self.assertNotIn("GEMINI_API_KEY", app)
        self.assertIn("no_llm=no_llm", app)
        self.assertIn("history=st.session_state.conversation_history[-4:]", app)
        self.assertIn("last_filters=st.session_state.last_filters", app)

    def test_explicit_blank_key_overrides_server_environment(self):
        tree = ast.parse(read("server.py"))
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_request_api_key"
        )
        class FakeOS:
            environ = {"GROQ_API_KEY": "server-secret"}
        namespace = {"os": FakeOS}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), namespace)
        resolve_key = namespace["_request_api_key"]
        self.assertEqual(resolve_key({}), "server-secret")
        self.assertEqual(resolve_key({"api_key": ""}), "")
        self.assertEqual(resolve_key({"api_key": "  "}), "")
        self.assertEqual(resolve_key({"api_key": "browser-key"}), "browser-key")
        self.assertEqual(resolve_key({"api_key": None}), "")

    def test_flask_local_defaults_and_no_wildcard_cors(self):
        server = read("server.py")
        self.assertIn('default="127.0.0.1"', server)
        self.assertNotIn("Access-Control-Allow-Origin", server)
        self.assertIn("SERVER_NO_LLM or chunks_only", server)
        self.assertIn("_client_filters(data.get(\"last_filters\"))", server)
        self.assertIn("brain.last_filters = previous_filters", server)

    def test_client_filter_validator(self):
        tree = ast.parse(read("server.py"))
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_client_filters"
        )
        namespace = {"re": re}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), namespace)
        validate = namespace["_client_filters"]
        self.assertEqual(validate(["2019", "370", "Lok Sabha"]),
                         ("2019", "370", "Lok Sabha"))
        self.assertEqual(validate([2018, "XXXVIII", "Rajya Sabha"]),
                         ("2018", "XXXVIII", "Rajya Sabha"))
        self.assertIsNone(validate([1999, "1", "Lok Sabha"]))
        self.assertIsNone(validate([2024, "not-a-number", "Lok Sabha"]))
        self.assertEqual(validate([2024, "1", "Other"]), ("2024", "1", None))
        self.assertIsNone(validate(None))

    def test_browser_client_keeps_its_own_followup_context(self):
        html = read("ui/index.html")
        self.assertIn("let lastFilters = null", html)
        self.assertIn("last_filters: lastFilters", html)
        self.assertIn("lastFilters = [data.filters.year", html)

    def test_facts_shortcut_uses_request_context(self):
        backend = read("step4_talk.py")
        self.assertIn("facts_reply(db, question, plan, active_filters, brain)", backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
