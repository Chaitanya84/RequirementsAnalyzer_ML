import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, AuthenticationError
from streamlit.testing.v1 import AppTest

from core.analyzer import ConflictAnalyzer
from core.ears_checker import EARSChecker
from core.graph_rag import GraphRAGEngine


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def authentication_error():
    request = httpx.Request("GET", "https://api.openai.com")
    response = httpx.Response(401, request=request)
    return AuthenticationError("Rejected", response=response, body=None)


class SessionKeyTests(unittest.TestCase):
    def test_key_entry_and_session_isolation(self):
        engines = []

        def make_engine(**kwargs):
            engine = MagicMock()
            engine.records = []
            engine.build_mermaid_graph.return_value = "graph TD"
            engines.append((kwargs["api_key"], engine))
            return engine

        with (
            patch("openai.OpenAI"),
            patch("core.graph_rag.GraphRAGEngine", side_effect=make_engine),
            patch("core.analyzer.ConflictAnalyzer"),
            patch("core.ears_checker.EARSChecker"),
        ):
            first = AppTest.from_file(str(APP_PATH)).run()
            first.button[0].click().run()
            self.assertIn("Enter your OpenAI API key", first.error[0].value)
            self.assertEqual(len(engines), 0)

            first.text_input[0].set_value(" first-key ")
            first.button[0].click().run()
            self.assertEqual(engines[0][0], "first-key")
            self.assertEqual(len(first.exception), 0)
            first.run()
            self.assertEqual(len(engines), 1)

            second = AppTest.from_file(str(APP_PATH)).run()
            second.text_input[0].set_value("second-key")
            second.button[0].click().run()
            self.assertEqual(engines[1][0], "second-key")
            self.assertIsNot(engines[0][1], engines[1][1])
            self.assertEqual(len(second.exception), 0)

    def test_invalid_key_or_verification_failure_never_initializes_services(self):
        failures = (
            authentication_error(),
            APIConnectionError(request=httpx.Request("GET", "https://api.openai.com")),
        )
        for failure in failures:
            with (
                self.subTest(failure=type(failure).__name__),
                patch("openai.OpenAI") as client,
                patch("core.graph_rag.GraphRAGEngine") as engine,
            ):
                client.return_value.models.list.side_effect = failure
                app = AppTest.from_file(str(APP_PATH)).run()
                app.text_input[0].set_value("invalid-key")
                app.button[0].click().run()
                self.assertEqual(len(app.tabs), 0)
                self.assertEqual(len(app.error), 1)
                self.assertEqual(len(app.exception), 0)
                engine.assert_not_called()
                client.return_value.close.assert_called_once()

    def test_authentication_failure_locks_every_api_action(self):
        for action in ("analyze", "ears", "add", "question", "graph"):
            with (
                self.subTest(action=action),
                patch("openai.OpenAI"),
                patch("core.graph_rag.GraphRAGEngine") as engine_class,
                patch("core.analyzer.ConflictAnalyzer") as analyzer_class,
                patch("core.ears_checker.EARSChecker") as ears_class,
            ):
                engine = engine_class.return_value
                engine.records = []
                engine.build_mermaid_graph.return_value = "graph TD"
                analyzer = analyzer_class.return_value
                analyzer.analyze.return_value = {"conflicts": [], "summary": "ok"}
                app = AppTest.from_file(str(APP_PATH)).run()
                app.text_input[0].set_value("previously-valid-key")
                app.button[0].click().run()
                self.assertEqual(len(app.tabs), 5)

                if action in ("analyze", "ears", "add"):
                    app.text_area[0].set_value("The system shall work.")
                    if action == "analyze":
                        analyzer.analyze.side_effect = authentication_error()
                        next(button for button in app.button if button.label == "Analyze Requirement").click()
                    elif action == "ears":
                        ears_class.return_value.validate.side_effect = authentication_error()
                        next(button for button in app.button if button.label == "Analyze Requirement").click()
                    else:
                        engine.add_requirements_from_text.side_effect = authentication_error()
                        next(button for button in app.button if button.label == "Add Requirement to Knowledge Base").click()
                elif action == "question":
                    engine.answer_question.side_effect = authentication_error()
                    app.chat_input[0].set_value("What are the requirements?")
                else:
                    engine.build_mermaid_graph.side_effect = authentication_error()
                    app.text_input[0].set_value("filter")

                app.run()
                self.assertEqual(len(app.tabs), 0)
                self.assertIn("rejected", app.error[0].value)
                self.assertEqual(app.text_input[0].label, "OpenAI API key")
                self.assertEqual(len(app.exception), 0)


class AuthenticationPropagationTests(unittest.TestCase):
    def test_core_does_not_convert_authentication_errors_to_fallbacks(self):
        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        engine.client = MagicMock()
        engine.model = "test"
        engine.embedding_model = "test"
        engine.allow_local_embedding_fallback = True
        engine.client.chat.completions.create.side_effect = authentication_error()
        engine.client.embeddings.create.side_effect = authentication_error()
        engine.retrieve = MagicMock(return_value={"context_text": "", "edges": []})
        engine._read_prompt = MagicMock(return_value="")

        with self.assertRaises(AuthenticationError):
            engine.extract_entities("requirement")
        with self.assertRaises(AuthenticationError):
            engine._embed_text("requirement")
        with self.assertRaises(AuthenticationError):
            engine.answer_question("question", "")
        engine.add_requirement = MagicMock(side_effect=authentication_error())
        with self.assertRaises(AuthenticationError):
            engine.add_requirements_from_text("The system shall work.")

        analyzer = ConflictAnalyzer.__new__(ConflictAnalyzer)
        analyzer.client = engine.client
        analyzer.model = "test"
        analyzer.graph_engine = MagicMock()
        analyzer.graph_engine.retrieve.return_value = {"records": []}
        with self.assertRaises(AuthenticationError):
            analyzer.analyze("requirement", "")

        checker = EARSChecker.__new__(EARSChecker)
        checker.client = engine.client
        checker.model = "test"
        with self.assertRaises(AuthenticationError):
            checker.validate("The system shall work.", "")

            first.sidebar.button[0].click().run()
            self.assertEqual(first.text_input[0].label, "OpenAI API key")
            self.assertEqual(first.text_input[0].value, "")
            self.assertEqual(len(first.exception), 0)
            first.text_input[0].set_value("replacement-key")
            first.button[0].click().run()
            self.assertEqual(engines[2][0], "replacement-key")
            second.run()
            self.assertEqual(len(engines), 3)
            self.assertEqual(len(second.exception), 0)


if __name__ == "__main__":
    unittest.main()
