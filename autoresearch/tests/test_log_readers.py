"""Tests for scripts/log_reader.py (US-515 phase 2).

Phase 2 drops the phase-1 dual-format shim — the wrapper now emits
native JSONL, so `_iter_legacy_events` and its regex synthesizers are
gone. These tests cover the remaining native-JSONL surface.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import autoresearch.log_reader as lr


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mixed_log_sample"
    / "autoresearch.jsonl"
)


class TestLogReader(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(FIXTURE.exists(), f"missing fixture: {FIXTURE}")

    def test_reads_all_events(self) -> None:
        events = list(lr.iter_events(jsonl_path=FIXTURE))
        self.assertGreaterEqual(len(events), 10)
        tss = [e["ts"] for e in events]
        self.assertEqual(tss, sorted(tss), "JSONL must be chronologically appended")

    def test_iteration_phase_event_exact_match(self) -> None:
        events = list(
            lr.iter_events(
                subsystem="wrapper", event="iteration.phase", jsonl_path=FIXTURE
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

    def test_event_exact_match_does_not_match_sub_events(self) -> None:
        events = list(
            lr.iter_events(
                subsystem="wrapper", event="iteration", jsonl_path=FIXTURE
            )
        )
        self.assertEqual(events, [], "event filter is exact-match only")

    def test_subsystem_dotted_prefix_matches_children(self) -> None:
        events = list(lr.iter_events(subsystem="retest", jsonl_path=FIXTURE))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["subsystem"], "retest.diagnose")

    def test_pipeline_failure(self) -> None:
        events = list(lr.iter_events(event="pipeline.failure", jsonl_path=FIXTURE))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["level"], "CRITICAL")
        self.assertIn("rc=2", events[0]["detail"])

    def test_filter_by_level(self) -> None:
        events = list(lr.iter_events(level="ERROR", jsonl_path=FIXTURE))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "retest.diagnose.traceback")

    def test_filter_by_since_strict_less_than(self) -> None:
        cutoff = "2026-04-18T10:01:00+00:00"
        events = list(lr.iter_events(since=cutoff, jsonl_path=FIXTURE))
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertGreaterEqual(e["ts"], cutoff)

    def test_missing_file_yields_empty(self) -> None:
        events = list(lr.iter_events(jsonl_path=Path("/tmp/does-not-exist.jsonl")))
        self.assertEqual(events, [])

    def test_malformed_line_skipped(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"ts": "2026-01-01T00:00:00+00:00", "event": "ok"}\n')
            f.write("not-json-at-all\n")
            f.write('{"ts": "2026-01-01T00:00:01+00:00", "event": "also-ok"}\n')
            tmp = Path(f.name)
        try:
            events = list(lr.iter_events(jsonl_path=tmp))
            self.assertEqual([e["event"] for e in events], ["ok", "also-ok"])
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
