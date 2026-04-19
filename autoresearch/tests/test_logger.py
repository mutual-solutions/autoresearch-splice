"""Unit tests for autoresearch/logger.py (US-515 phase 1 / US-516 post-reorg)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

import autoresearch.logger as lg


class LoggerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_override = os.environ.get("OMC_LOG_OVERRIDE")
        self._prev_disabled = os.environ.get("OMC_LOGGER_DISABLED")
        self.tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
        self.log_path = self.tmp_dir / f"test_logger_{os.getpid()}_{id(self)}.jsonl"
        if self.log_path.exists():
            self.log_path.unlink()
        os.environ["OMC_LOG_OVERRIDE"] = str(self.log_path)
        os.environ.pop("OMC_LOGGER_DISABLED", None)
        lg._CACHE.clear()

    def tearDown(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()
        parent = self.log_path.parent
        for sib in parent.glob(f"{self.log_path.stem}.*{self.log_path.suffix}"):
            sib.unlink(missing_ok=True)
        if self._prev_override is None:
            os.environ.pop("OMC_LOG_OVERRIDE", None)
        else:
            os.environ["OMC_LOG_OVERRIDE"] = self._prev_override
        if self._prev_disabled is None:
            os.environ.pop("OMC_LOGGER_DISABLED", None)
        else:
            os.environ["OMC_LOGGER_DISABLED"] = self._prev_disabled
        lg._CACHE.clear()

    def _read_records(self):
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class TestEmitSchema(LoggerTestBase):
    def test_emit_schema(self) -> None:
        log = lg.get_logger("detector.gbm")
        log.emit("INFO", "diag.gbm.dedupe", before=120, after=80)
        records = self._read_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        for key in ("schema_version", "ts", "level", "subsystem", "event"):
            self.assertIn(key, rec)
        self.assertEqual(rec["schema_version"], 1)
        self.assertEqual(rec["level"], "INFO")
        self.assertEqual(rec["subsystem"], "detector.gbm")
        self.assertEqual(rec["event"], "diag.gbm.dedupe")
        self.assertEqual(rec["before"], 120)
        self.assertEqual(rec["after"], 80)
        self.assertEqual(rec["claude_visible"], False)

    def test_get_logger_caches(self) -> None:
        a = lg.get_logger("eval.run")
        b = lg.get_logger("eval.run")
        self.assertIs(a, b)

    def test_invalid_level_coerced_to_info(self) -> None:
        log = lg.get_logger("t")
        log.emit("NONSENSE", "diag.t.x")
        rec = self._read_records()[0]
        self.assertEqual(rec["level"], "INFO")


class TestRedaction(LoggerTestBase):
    def test_redaction_forced_on_retest_diagnose(self) -> None:
        log = lg.get_logger("retest.diagnose")
        log.emit(
            "ERROR",
            "retest.diagnose.traceback",
            detail="combined=0.47 crashed in hotelling",
            other="splice_f1=0.34 plateau",
        )
        rec = self._read_records()[0]
        self.assertIn("<REDACTED>", rec["detail"])
        self.assertNotIn("0.47", rec["detail"])
        self.assertIn("<REDACTED>", rec["other"])
        self.assertNotIn("0.34", rec["other"])

    def test_redaction_on_explicit_oracle_sensitive(self) -> None:
        log = lg.get_logger("wrapper")
        log.emit(
            "INFO",
            "note.appended",
            summary="combined_mean=0.468 combined_singing=0.24",
            oracle_sensitive=True,
        )
        rec = self._read_records()[0]
        self.assertIn("<REDACTED>", rec["summary"])

    def test_no_redaction_by_default(self) -> None:
        log = lg.get_logger("wrapper")
        log.emit("INFO", "eval.run.complete", summary="combined=0.475 elapsed=180s")
        rec = self._read_records()[0]
        self.assertIn("0.475", rec["summary"])
        self.assertNotIn("<REDACTED>", rec["summary"])

    def test_redaction_preserves_non_string_values(self) -> None:
        log = lg.get_logger("retest.diagnose")
        log.emit(
            "ERROR",
            "retest.diagnose.per_domain_error",
            combined_value=0.47,
            iter_count=37,
        )
        rec = self._read_records()[0]
        self.assertEqual(rec["combined_value"], 0.47)
        self.assertEqual(rec["iter_count"], 37)


class TestRendererCreatesMissingFile(LoggerTestBase):
    def test_renderer_creates_missing_file(self) -> None:
        log = lg.get_logger("wrapper")
        log.emit("INFO", "wrapper.iteration.start", iter=42, claude_visible=True)
        log.emit("INFO", "diag.hidden.event", x=1)
        target = self.tmp_dir / f"test_renderer_{os.getpid()}" / "notes.md"
        if target.exists():
            target.unlink()
        if target.parent.exists():
            target.parent.rmdir()
        self.assertFalse(target.parent.exists())
        count = lg.render_claude_visible_events(
            target=target, jsonl_path=self.log_path
        )
        self.assertEqual(count, 1)
        self.assertTrue(target.exists())
        body = target.read_text()
        self.assertIn("wrapper.iteration.start", body)
        self.assertNotIn("diag.hidden.event", body)
        target.unlink()
        target.parent.rmdir()

    def test_renderer_appends_not_overwrites(self) -> None:
        target = self.tmp_dir / f"test_renderer_append_{os.getpid()}.md"
        target.write_text("# Existing content\n")
        log = lg.get_logger("wrapper")
        log.emit("INFO", "wrapper.iteration.start", iter=1, claude_visible=True)
        lg.render_claude_visible_events(target=target, jsonl_path=self.log_path)
        body = target.read_text()
        self.assertIn("# Existing content", body)
        self.assertIn("wrapper.iteration.start", body)
        target.unlink()

    def test_renderer_empty_on_no_claude_visible(self) -> None:
        log = lg.get_logger("wrapper")
        log.emit("INFO", "diag.any.x", k=1)
        log.emit("INFO", "diag.any.y", k=2)
        target = self.tmp_dir / f"test_renderer_empty_{os.getpid()}.md"
        target.unlink(missing_ok=True)
        count = lg.render_claude_visible_events(
            target=target, jsonl_path=self.log_path
        )
        self.assertEqual(count, 0)


class TestRotation(LoggerTestBase):
    def test_rotation(self) -> None:
        log = lg.get_logger("wrapper")
        with patch.object(lg, "MAX_SIZE_BYTES", 512):
            for i in range(200):
                log.emit("INFO", "wrapper.noise", i=i, payload="x" * 40)
        rolls = sorted(
            self.log_path.parent.glob(f"{self.log_path.stem}.*{self.log_path.suffix}")
        )
        self.assertGreaterEqual(len(rolls), 1)
        self.assertTrue(self.log_path.exists())

    def test_rotation_prunes_to_max_rolls(self) -> None:
        log = lg.get_logger("wrapper")
        with patch.object(lg, "MAX_SIZE_BYTES", 256), patch.object(lg, "MAX_ROLLS", 3):
            for i in range(400):
                log.emit("INFO", "wrapper.noise", i=i, payload="x" * 40)
        rolls = sorted(
            self.log_path.parent.glob(f"{self.log_path.stem}.*{self.log_path.suffix}")
        )
        self.assertLessEqual(len(rolls), 3)


class TestLoggerDisabledEnv(LoggerTestBase):
    def test_logger_disabled_env(self) -> None:
        os.environ["OMC_LOGGER_DISABLED"] = "1"
        log = lg.get_logger("wrapper")
        log.emit("INFO", "diag.anything", x=1)
        self.assertFalse(self.log_path.exists())

    def test_logger_re_enables_when_unset(self) -> None:
        os.environ["OMC_LOGGER_DISABLED"] = "1"
        log = lg.get_logger("wrapper")
        log.emit("INFO", "diag.a", x=1)
        self.assertFalse(self.log_path.exists())
        os.environ.pop("OMC_LOGGER_DISABLED")
        log.emit("INFO", "diag.b", x=2)
        self.assertTrue(self.log_path.exists())
        recs = self._read_records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["event"], "diag.b")


if __name__ == "__main__":
    unittest.main()
