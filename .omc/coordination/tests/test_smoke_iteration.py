#!/usr/bin/env python3
"""US-515 phase-1 pre-merge smoke gate.

Python-only stub: simulates one iteration's worth of logger emission into
a temp fixture, then runs all four phase-1 gates against it.

DOES NOT invoke run_autoresearch.sh, evaluate.py, or any real dataset.
Runs in <5 seconds. PR is not mergeable if any of the four gates fails.

Gates:
  1. Fixture emission — real `emit()` calls (no OMC_LOGGER_DISABLED)
  2. Parse gate       — validate_logs.py --parse exits 0
  3. Reader gate      — phase_stats.py + dashboard.py produce non-empty output
  4. Redaction gate   — retest.diagnose.* records contain <REDACTED>, not 0.47

Usage (exact command; NOT `python -m omc.*` — `.omc` has a leading dot so
packaging under that name is impossible):

  uv run python .omc/coordination/tests/test_smoke_iteration.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".omc" / "coordination"))

import logger as lg  # noqa: E402


class SmokeIteration(unittest.TestCase):
    """One-iteration end-to-end smoke covering all phase-1 gates."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="us515_smoke_")
        self.root = Path(self._tmp.name)
        self.jsonl = self.root / "autoresearch.jsonl"
        self.legacy = self.root / "autoresearch.log"
        self._prev_override = os.environ.get("OMC_LOG_OVERRIDE")
        self._prev_disabled = os.environ.get("OMC_LOGGER_DISABLED")
        os.environ["OMC_LOG_OVERRIDE"] = str(self.jsonl)
        os.environ.pop("OMC_LOGGER_DISABLED", None)
        lg._CACHE.clear()

    def tearDown(self) -> None:
        if self._prev_override is None:
            os.environ.pop("OMC_LOG_OVERRIDE", None)
        else:
            os.environ["OMC_LOG_OVERRIDE"] = self._prev_override
        if self._prev_disabled is None:
            os.environ.pop("OMC_LOGGER_DISABLED", None)
        else:
            os.environ["OMC_LOGGER_DISABLED"] = self._prev_disabled
        lg._CACHE.clear()
        self._tmp.cleanup()

    def _emit_iteration_fixture(self) -> None:
        """Gate 1: emit a representative one-iteration set of events."""
        wrapper = lg.get_logger("wrapper")
        eval_ = lg.get_logger("eval")
        classifier = lg.get_logger("classifier.train")
        retest = lg.get_logger("retest.diagnose")
        detector = lg.get_logger("detector.gbm")

        wrapper.emit("INFO", "wrapper.iteration.start", iter="abc1234",
                     discards_since_last_keep=4, claude_visible=True)
        classifier.emit("INFO", "classifier.retrain.trigger",
                        reason="features.py changed")
        classifier.emit("INFO", "classifier.retrain.complete",
                        samples=7200, elapsed_s=172)
        detector.emit("INFO", "diag.gbm.dedupe", before=120, after=80)
        eval_.emit("INFO", "eval.run", datasets=3)
        eval_.emit("INFO", "eval.dataset.result", ds_id="singing",
                   combined=0.24, splice_f1=0.34)
        eval_.emit("INFO", "eval.dataset.result", ds_id="korean",
                   combined=0.48, splice_f1=0.55)
        eval_.emit("INFO", "eval.dataset.result", ds_id="english",
                   combined=0.66, splice_f1=0.71)
        eval_.emit("INFO", "eval.aggregate", combined_mean=0.46,
                   combined_min=0.24)
        retest.emit("ERROR", "retest.diagnose.traceback",
                    detail="combined=0.47 crashed in hotelling")
        wrapper.emit("INFO", "note.appended", file=".omc/research_notes.md",
                     iter="abc1234")

        self._write_legacy_fixture()
        self.assertTrue(self.jsonl.exists(), "JSONL fixture not written")
        self.assertTrue(self.legacy.exists(), "legacy fixture not written")

    def _write_legacy_fixture(self) -> None:
        """Paired legacy-line fixture — wrapper-shim events need these."""
        self.legacy.write_text(
            "2026-04-18T10:00:05+00:00 Starting iteration "
            "(consecutive discards: 4)\n"
            "2026-04-18T10:00:06+00:00 PHASE: claude start iter=abc1234\n"
            "2026-04-18T10:00:16+00:00 PHASE: claude end\n"
            "2026-04-18T10:00:17+00:00 PHASE: eval start\n"
            "2026-04-18T10:00:35+00:00 PHASE: eval end\n"
            "2026-04-18T10:00:40+00:00 ITER_SUMMARY iter=abc1234 "
            "status=discard total=38 claude=10 retrain=- eval=19 verify=3 note=1\n"
            "2026-04-18T10:00:41+00:00 Iteration: discard (discards: 5/50)\n"
        )

    def test_gate_1_fixture_emission(self) -> None:
        self._emit_iteration_fixture()
        with open(self.jsonl, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        self.assertGreaterEqual(len(lines), 10, "expected >=10 JSONL records")

    def test_gate_2_parse(self) -> None:
        self._emit_iteration_fixture()
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "validate_logs.py"),
             "--parse", str(self.jsonl)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        self.assertEqual(result.returncode, 0,
                         f"--parse failed: stderr={result.stderr}")

    def test_gate_3_readers_non_empty(self) -> None:
        self._emit_iteration_fixture()
        env = os.environ.copy()
        env["OMC_LOG_OVERRIDE"] = str(self.jsonl)

        r_phase = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "phase_stats.py"),
             "--log", str(self.legacy)],
            capture_output=True, text=True, cwd=str(REPO), env=env,
        )
        self.assertEqual(r_phase.returncode, 0,
                         f"phase_stats.py failed: stderr={r_phase.stderr}")
        self.assertTrue(r_phase.stdout.strip(),
                        "phase_stats.py stdout is empty")

        r_dash = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "dashboard.py"), "--help"],
            capture_output=True, text=True, cwd=str(REPO), env=env,
        )
        self.assertEqual(r_dash.returncode, 0,
                         f"dashboard.py --help failed: stderr={r_dash.stderr}")
        self.assertTrue(r_dash.stdout.strip(),
                        "dashboard.py --help stdout is empty")

    def test_gate_4_redaction(self) -> None:
        self._emit_iteration_fixture()
        with open(self.jsonl, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        traceback = [
            r for r in records if r["event"] == "retest.diagnose.traceback"
        ]
        self.assertEqual(len(traceback), 1,
                         "expected exactly one retest.diagnose.traceback")
        detail = traceback[0]["detail"]
        self.assertIn("<REDACTED>", detail,
                      f"expected <REDACTED>; got {detail!r}")
        self.assertNotIn("0.47", detail,
                         f"oracle value 0.47 leaked: {detail!r}")


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(SmokeIteration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
