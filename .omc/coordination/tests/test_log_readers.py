"""Tests for scripts/log_reader.py (US-515 phase 1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import log_reader as lr  # noqa: E402


class TestDualFormatMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).resolve().parent / "fixtures" / "mixed_log_sample"
        self.jsonl = self.fixture_dir / "autoresearch.jsonl"
        self.legacy = self.fixture_dir / "autoresearch.log"
        self.assertTrue(self.jsonl.exists())
        self.assertTrue(self.legacy.exists())

    def test_dual_format_merge(self) -> None:
        events = list(lr.iter_events(jsonl_path=self.jsonl, legacy_path=self.legacy))
        self.assertGreaterEqual(len(events), 10)
        tss = [e["ts"] for e in events]
        self.assertEqual(tss, sorted(tss))
        sources = {e.get("source", "jsonl") for e in events}
        self.assertIn("legacy", sources)

    def test_iter_summary_synthesized_as_wrapper_iteration_phase(self) -> None:
        events = list(
            lr.iter_events(
                subsystem="wrapper",
                event="iteration.phase",
                jsonl_path=self.jsonl,
                legacy_path=self.legacy,
            )
        )
        self.assertEqual(len(events), 2)
        e0 = events[0]
        self.assertEqual(e0["subsystem"], "wrapper")
        self.assertEqual(e0["event"], "iteration.phase")
        self.assertEqual(e0["iter"], "abc1234")
        self.assertEqual(e0["status"], "discard")
        self.assertEqual(e0["claude"], "10")
        self.assertEqual(e0["eval"], "19")

    def test_phase_step_synthesized(self) -> None:
        events = list(
            lr.iter_events(
                subsystem="wrapper",
                event="iteration.phase.step",
                jsonl_path=self.jsonl,
                legacy_path=self.legacy,
            )
        )
        actions = {(e["phase"], e["action"]) for e in events}
        self.assertIn(("claude", "start"), actions)
        self.assertIn(("claude", "end"), actions)
        self.assertIn(("eval", "start"), actions)
        self.assertIn(("eval", "end"), actions)

    def test_pipeline_failure_synthesized(self) -> None:
        events = list(
            lr.iter_events(
                event="pipeline.failure",
                jsonl_path=self.jsonl,
                legacy_path=self.legacy,
            )
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["level"], "CRITICAL")
        self.assertIn("rc=2", events[0]["detail"])

    def test_filter_by_level(self) -> None:
        events = list(
            lr.iter_events(level="ERROR", jsonl_path=self.jsonl, legacy_path=self.legacy)
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "retest.diagnose.traceback")

    def test_filter_by_since(self) -> None:
        events = list(
            lr.iter_events(
                since="2026-04-18T10:01:00+00:00",
                jsonl_path=self.jsonl,
                legacy_path=self.legacy,
            )
        )
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertGreater(e["ts"], "2026-04-18T10:01:00+00:00")

    def test_missing_jsonl_yields_only_legacy(self) -> None:
        events = list(
            lr.iter_events(
                jsonl_path=Path("/tmp/does-not-exist.jsonl"),
                legacy_path=self.legacy,
            )
        )
        self.assertGreater(len(events), 0)
        self.assertTrue(all(e.get("source") == "legacy" for e in events))

    def test_missing_both_yields_empty(self) -> None:
        events = list(
            lr.iter_events(
                jsonl_path=Path("/tmp/nope.jsonl"),
                legacy_path=Path("/tmp/also-nope.log"),
            )
        )
        self.assertEqual(events, [])

    def test_dotted_prefix_subsystem_match(self) -> None:
        events = list(
            lr.iter_events(subsystem="retest", jsonl_path=self.jsonl, legacy_path=self.legacy)
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["subsystem"], "retest.diagnose")


if __name__ == "__main__":
    unittest.main()
