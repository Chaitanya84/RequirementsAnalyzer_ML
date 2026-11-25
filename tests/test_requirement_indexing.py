import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from core.graph_rag import GraphRAGEngine


CONTENT = """# Requirements

### Power
* **CR-001:** The system shall start.
* **CR-002:** The system must stop.
  The stop must be safe.

### Hardware
- **HWR-001**: The hardware shall monitor power.

| Customer ID | System ID | Software ID | Hardware ID |
| :--- | :--- | :--- | :--- |
| CR-001 | SYSR-001 | SWR-001 | HWR-001 |
"""


class RequirementIndexingTests(unittest.TestCase):
    def test_extracts_each_numbered_bullet_and_preserves_ids(self):
        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        self.assertEqual(
            engine._extract_requirements(CONTENT),
            [
                ("CR-001", "The system shall start."),
                ("CR-002", "The system must stop. The stop must be safe."),
                ("HWR-001", "The hardware shall monitor power."),
            ],
        )

    def test_existing_table_and_paragraph_formats_still_work(self):
        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        table = "| REQ-001 | High | The system shall monitor the input continuously. |"
        self.assertEqual(
            engine._extract_requirements(table),
            [("REQ-001", "The system shall monitor the input continuously.")],
        )
        paragraph = "The system shall monitor the input continuously."
        self.assertEqual(engine._extract_requirements(paragraph), [(None, paragraph)])

    def test_real_document_if_available(self):
        path = Path(__file__).resolve().parents[1] / "input" / "ALLReq.md"
        if not path.exists():
            self.skipTest("Local input document is not included in the repository")
        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        requirements = engine._extract_requirements(path.read_text(encoding="utf-8"))
        self.assertEqual(len(requirements), 97)
        self.assertEqual(len({req_id for req_id, _ in requirements}), 97)
        self.assertIn("must never", dict(requirements)["CR-016"])

    def test_real_document_migrates_all_old_groups_if_available(self):
        path = Path(__file__).resolve().parents[1] / "input" / "ALLReq.md"
        if not path.exists():
            self.skipTest("Local input document is not included in the repository")
        content = path.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("core.graph_rag.OpenAI"),
            patch.object(GraphRAGEngine, "extract_entities", return_value={}),
            patch.object(GraphRAGEngine, "_embed_text", return_value=np.array([1.0, 0.0], dtype=np.float32)),
        ):
            root = Path(folder)
            engine = GraphRAGEngine("test", "test", str(root / "input"), "prompts", storage_path=str(root / "store"))
            for paragraph in re.split(r"\n\s*\n", content):
                if re.search(r"\bshall\b", paragraph, re.IGNORECASE) and len(paragraph.strip()) >= 25:
                    engine.add_requirement(paragraph, source="ALLReq.md")
            self.assertEqual(len(engine.records), 18)
            result = engine.add_requirements_from_text(content, source="ALLReq.md")
            self.assertEqual((result["parsed_count"], result["added_count"], result["replaced_group_count"]), (97, 97, 18))
            self.assertEqual(len(engine.records), 97)
            self.assertEqual(engine.index.ntotal, 97)

    def test_retry_replaces_groups_only_when_all_entries_are_indexed(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("core.graph_rag.OpenAI"),
            patch.object(GraphRAGEngine, "extract_entities", return_value={}),
            patch.object(GraphRAGEngine, "_embed_text", return_value=np.array([1.0, 0.0], dtype=np.float32)),
        ):
            root = Path(folder)
            engine = GraphRAGEngine("test", "test", str(root / "input"), "prompts", storage_path=str(root / "store"))
            legacy = CONTENT.split("\n\n")[1]
            engine.add_requirement(legacy, source="ALLReq.md")
            engine.add_requirement("Unrelated requirement shall remain.", source="other")
            original_add = engine.add_requirement

            def fail_one(text, source="user", source_req_id=None):
                if source_req_id == "CR-002":
                    raise RuntimeError("embedding failed")
                return original_add(text, source, source_req_id)

            with patch.object(engine, "add_requirement", side_effect=fail_one):
                first = engine.add_requirements_from_text(CONTENT, source="ALLReq.md")
            self.assertEqual((first["added_count"], first["failed_count"], first["replaced_group_count"]), (2, 1, 0))
            self.assertTrue(any(record.text == engine._normalize_requirement_text(legacy) for record in engine.records))

            second = engine.add_requirements_from_text(CONTENT, source="ALLReq.md")
            self.assertEqual((second["added_count"], second["duplicate_count"], second["replaced_group_count"]), (1, 2, 1))
            self.assertEqual(len(engine.records), 4)
            self.assertEqual(engine.index.ntotal, 4)
            self.assertEqual(len(set(engine.record_ids)), 4)
            self.assertIn("Unrelated requirement shall remain.", [record.text for record in engine.records])

            reloaded = GraphRAGEngine("test", "test", str(root / "input"), "prompts", storage_path=str(root / "store"))
            self.assertEqual(len(reloaded.records), 4)
            self.assertEqual(reloaded.index.ntotal, 4)
            self.assertEqual({r.source_req_id for r in reloaded.records if r.source_req_id}, {"CR-001", "CR-002", "HWR-001"})
            self.assertEqual(len(reloaded.retrieve("power")["records"]), 4)
            third = reloaded.add_requirements_from_text(CONTENT, source="ALLReq.md")
            self.assertEqual((third["added_count"], third["duplicate_count"], third["replaced_group_count"]), (0, 3, 0))


class IngestionSummaryTests(unittest.TestCase):
    def test_partial_failure_summary_remains_visible_after_rerun(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with (
            patch("openai.OpenAI"),
            patch("core.graph_rag.GraphRAGEngine") as engine_class,
            patch("core.analyzer.ConflictAnalyzer"),
            patch("core.ears_checker.EARSChecker"),
        ):
            engine = engine_class.return_value
            engine.records = []
            engine.build_mermaid_graph.return_value = "graph TD"
            engine.add_requirements_from_text.return_value = {
                "parsed_count": 2,
                "added_count": 0,
                "duplicate_count": 1,
                "failed_count": 1,
                "replaced_group_count": 0,
                "errors": ["item 2: embedding failed"],
            }
            app = AppTest.from_file(str(app_path)).run()
            app.text_input[0].set_value("test-key")
            app.button[0].click().run()
            app.text_area[0].set_value(CONTENT)
            next(button for button in app.button if button.label == "Add Requirement to Knowledge Base").click().run()
            self.assertTrue(any("parsed=2" in text.value and "failed=1" in text.value for text in app.markdown))
            self.assertTrue(any("could not be indexed" in warning.value for warning in app.warning))
            self.assertTrue(any("embedding failed" in text.value for text in app.markdown))
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
