"""Tests for US-517 --maintain subcommand.

12 cases covering: backlog parse/roundtrip, fuzzy match, bullet extraction,
crash classifier, backlog flow, and spec-drafting.

Uses importlib.util.spec_from_file_location (same pattern as test_retest.py)
so the module is loaded hermetically without package-discovery side effects.
Tests use tempfile.TemporaryDirectory for isolation; real .omc/ is never touched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_AGENT_PATH = REPO_ROOT / "autoresearch" / "supervisor_agent.py"
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "maintain"


def _load_supervisor_agent():
    """Import supervisor_agent.py as an ephemeral module for each test.

    Sets OMC_LOGGER_DISABLED=1 so emit() is a no-op during tests.
    """
    os.environ.setdefault("OMC_LOGGER_DISABLED", "1")
    spec = importlib.util.spec_from_file_location(
        f"_supervisor_agent_test_{id(object())}", SUPERVISOR_AGENT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


BOOTSTRAP_BACKLOG = (FIXTURES_DIR / "backlog_bootstrap.md").read_text(encoding="utf-8")


class TestParseBacklogRoundtrip(unittest.TestCase):
    """test_parse_backlog_roundtrip: parse + serialize produces identical output."""

    def test_parse_backlog_roundtrip(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            backlog_path = Path(tmpdir) / "backlog.md"
            backlog_path.write_text(BOOTSTRAP_BACKLOG, encoding="utf-8")

            header, entries = mod._parse_backlog(backlog_path)
            mod._serialize_backlog(header, entries, backlog_path)
            result = backlog_path.read_text(encoding="utf-8")

        # Round-trip: content must be semantically equivalent
        # (we allow trailing-whitespace differences per the plan)
        original_stripped = "\n".join(line.rstrip() for line in BOOTSTRAP_BACKLOG.splitlines())
        result_stripped = "\n".join(line.rstrip() for line in result.splitlines())
        self.assertEqual(original_stripped.strip(), result_stripped.strip())


class TestFuzzyMatchDedupPositive(unittest.TestCase):
    """test_fuzzy_match_dedup_positive: near-verbatim excerpts match >= 0.5."""

    def test_fuzzy_match_dedup_positive(self):
        mod = _load_supervisor_agent()
        entries = [
            {
                "id": "shap-delta-block",
                "excerpt": "Surface a SHAP-DELTA block in CURRENT STATE for the most recent KEPT classifier vs prior KEPT classifier.",
            }
        ]
        # Near-verbatim variant — should match
        needle = "Surface a SHAP-DELTA block in CURRENT STATE for kept classifier vs prior kept classifier."
        match = mod._fuzzy_match(needle, entries)
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], "shap-delta-block")


class TestFuzzyMatchDedupNegative(unittest.TestCase):
    """test_fuzzy_match_dedup_negative: unrelated excerpts score < 0.5."""

    def test_fuzzy_match_dedup_negative(self):
        mod = _load_supervisor_agent()
        entries = [
            {
                "id": "shap-delta-block",
                "excerpt": "Surface a SHAP-DELTA block in CURRENT STATE for the most recent KEPT classifier.",
            }
        ]
        needle = "A completely different request about local file statistics and audio normalization."
        match = mod._fuzzy_match(needle, entries)
        self.assertIsNone(match)


class TestExtractBulletsSkipsBeforeSinceSha(unittest.TestCase):
    """test_extract_bullets_skips_entries_before_since_sha: older SHAs excluded."""

    def test_extract_bullets_skips_entries_before_since_sha(self):
        mod = _load_supervisor_agent()
        # The fixture has 3 entries: abc1234, def5678, ghi9012
        # We mock _sha_commit_time to return deterministic times.
        _orig = mod._sha_commit_time

        def _mock_commit_time(sha):
            times = {"abc1234": 1000, "def5678": 2000, "9abc012": 3000}
            return times.get(sha)

        mod._sha_commit_time = _mock_commit_time
        try:
            # since_sha = def5678 (time=2000). Only 9abc012 (time=3000) should pass.
            bullets = mod._extract_enhancement_bullets(
                FIXTURES_DIR / "research_notes_sample.md",
                since_sha="def5678",
            )
        finally:
            mod._sha_commit_time = _orig

        # Only 9abc012 bullets should be present
        shas = [sha for sha, _ in bullets]
        self.assertNotIn("abc1234", shas)
        self.assertNotIn("def5678", shas)
        self.assertIn("9abc012", shas)


class TestFreshBacklogNoNewBullets(unittest.TestCase):
    """test_fresh_backlog_no_new_bullets: clean backlog + no new notes -> exit 0, no-op."""

    def test_fresh_backlog_no_new_bullets(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Set up backlog
            backlog_path = tmpdir / "enhancement-backlog.md"
            backlog_path.write_text(BOOTSTRAP_BACKLOG, encoding="utf-8")
            # Empty research notes
            notes_path = tmpdir / "research_notes.md"
            notes_path.write_text("", encoding="utf-8")

            # Override paths
            orig_backlog = mod._BACKLOG_PATH
            orig_notes = mod._RESEARCH_NOTES_PATH
            orig_retest = mod._RETEST_SENTINEL
            orig_disabled = mod._MAINTAIN_DISABLED_SENTINEL
            orig_counter = mod._CRASH_COUNTER_PATH
            orig_specs = mod._SPECS_DIR
            orig_last_eval = mod.REPO_ROOT / ".omc" / "last_eval.log"

            mod._BACKLOG_PATH = backlog_path
            mod._RESEARCH_NOTES_PATH = notes_path
            mod._RETEST_SENTINEL = tmpdir / "retest-in-progress"
            mod._MAINTAIN_DISABLED_SENTINEL = tmpdir / "maintainer-disabled"
            mod._CRASH_COUNTER_PATH = tmpdir / "crash-counter.txt"
            mod._SPECS_DIR = tmpdir / "specs"
            # Patch REPO_ROOT so last_eval.log doesn't exist
            mod.REPO_ROOT = tmpdir

            try:
                rc = mod.run_maintain("periodic")
            finally:
                mod._BACKLOG_PATH = orig_backlog
                mod._RESEARCH_NOTES_PATH = orig_notes
                mod._RETEST_SENTINEL = orig_retest
                mod._MAINTAIN_DISABLED_SENTINEL = orig_disabled
                mod._CRASH_COUNTER_PATH = orig_counter
                mod._SPECS_DIR = orig_specs
                mod.REPO_ROOT = orig_last_eval.parent.parent

            self.assertEqual(rc, 0)
            # Backlog should be unchanged (no new entries)
            result_text = backlog_path.read_text(encoding="utf-8")
            original_stripped = "\n".join(l.rstrip() for l in BOOTSTRAP_BACKLOG.splitlines())
            result_stripped = "\n".join(l.rstrip() for l in result_text.splitlines())
            self.assertEqual(original_stripped.strip(), result_stripped.strip())


class TestNewBulletMatchesExistingBumpsCount(unittest.TestCase):
    """test_new_bullet_matches_existing_entry_bumps_count: fuzzy match -> request_count++."""

    def test_new_bullet_matches_existing_entry_bumps_count(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            backlog_path = tmpdir / "backlog.md"
            backlog_path.write_text(BOOTSTRAP_BACKLOG, encoding="utf-8")

            # Research notes with a near-verbatim variant of an existing entry
            notes = (
                "## 2026-04-18T17:00:00+09:00 — abc9999 (discard, combined=0.400000)\n"
                "subject: test\n\n"
                "(e) Wrapper enhancements. Surface a SHAP-DELTA block in CURRENT STATE "
                "for most recent KEPT classifier vs prior.\n"
            )
            notes_path = tmpdir / "research_notes.md"
            notes_path.write_text(notes, encoding="utf-8")

            _orig_time = mod._sha_commit_time
            def _mock_time(sha):
                # Make abc9999 newer than def5678 (last_seen in bootstrap for shap-delta-block)
                return {"abc9999": 9999999, "def5678": 1000, "abc1234": 500}.get(sha)
            mod._sha_commit_time = _mock_time

            orig_backlog = mod._BACKLOG_PATH
            orig_notes = mod._RESEARCH_NOTES_PATH
            orig_retest = mod._RETEST_SENTINEL
            orig_disabled = mod._MAINTAIN_DISABLED_SENTINEL
            orig_counter = mod._CRASH_COUNTER_PATH
            orig_specs = mod._SPECS_DIR
            orig_root = mod.REPO_ROOT

            mod._BACKLOG_PATH = backlog_path
            mod._RESEARCH_NOTES_PATH = notes_path
            mod._RETEST_SENTINEL = tmpdir / "retest-in-progress"
            mod._MAINTAIN_DISABLED_SENTINEL = tmpdir / "maintainer-disabled"
            mod._CRASH_COUNTER_PATH = tmpdir / "crash-counter.txt"
            mod._SPECS_DIR = tmpdir / "specs"
            mod.REPO_ROOT = tmpdir

            try:
                rc = mod.run_maintain("periodic")
            finally:
                mod._sha_commit_time = _orig_time
                mod._BACKLOG_PATH = orig_backlog
                mod._RESEARCH_NOTES_PATH = orig_notes
                mod._RETEST_SENTINEL = orig_retest
                mod._MAINTAIN_DISABLED_SENTINEL = orig_disabled
                mod._CRASH_COUNTER_PATH = orig_counter
                mod._SPECS_DIR = orig_specs
                mod.REPO_ROOT = orig_root

            self.assertEqual(rc, 0)
            _, entries = mod._parse_backlog(backlog_path)
            shap_entry = next((e for e in entries if e.get("id") == "shap-delta-block"), None)
            self.assertIsNotNone(shap_entry)
            count = int(shap_entry["request_count"])
            self.assertGreater(count, 2)  # was 2, should be 3+
            self.assertEqual(shap_entry["last_seen"], "abc9999")


class TestNewBulletUnmatchedAppendsEntry(unittest.TestCase):
    """test_new_bullet_unmatched_appends_entry: novel bullet creates new H2."""

    def test_new_bullet_unmatched_appends_entry(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            backlog_path = tmpdir / "backlog.md"
            backlog_path.write_text(BOOTSTRAP_BACKLOG, encoding="utf-8")

            notes = (
                "## 2026-04-18T17:00:00+09:00 — abc9999 (discard, combined=0.400000)\n"
                "subject: test\n\n"
                "(e) Wrapper enhancements. A completely novel request about audio fingerprinting "
                "and local novelty statistics for chunk-level comparison.\n"
            )
            notes_path = tmpdir / "research_notes.md"
            notes_path.write_text(notes, encoding="utf-8")

            _orig_time = mod._sha_commit_time
            def _mock_time(sha):
                return {"abc9999": 9999999, "def5678": 1000, "abc1234": 500}.get(sha)
            mod._sha_commit_time = _mock_time

            orig_backlog = mod._BACKLOG_PATH
            orig_notes = mod._RESEARCH_NOTES_PATH
            orig_retest = mod._RETEST_SENTINEL
            orig_disabled = mod._MAINTAIN_DISABLED_SENTINEL
            orig_counter = mod._CRASH_COUNTER_PATH
            orig_specs = mod._SPECS_DIR
            orig_root = mod.REPO_ROOT

            mod._BACKLOG_PATH = backlog_path
            mod._RESEARCH_NOTES_PATH = notes_path
            mod._RETEST_SENTINEL = tmpdir / "retest-in-progress"
            mod._MAINTAIN_DISABLED_SENTINEL = tmpdir / "maintainer-disabled"
            mod._CRASH_COUNTER_PATH = tmpdir / "crash-counter.txt"
            mod._SPECS_DIR = tmpdir / "specs"
            mod.REPO_ROOT = tmpdir

            _, entries_before = mod._parse_backlog(backlog_path)
            count_before = len(entries_before)

            try:
                rc = mod.run_maintain("periodic")
            finally:
                mod._sha_commit_time = _orig_time
                mod._BACKLOG_PATH = orig_backlog
                mod._RESEARCH_NOTES_PATH = orig_notes
                mod._RETEST_SENTINEL = orig_retest
                mod._MAINTAIN_DISABLED_SENTINEL = orig_disabled
                mod._CRASH_COUNTER_PATH = orig_counter
                mod._SPECS_DIR = orig_specs
                mod.REPO_ROOT = orig_root

            self.assertEqual(rc, 0)
            _, entries_after = mod._parse_backlog(backlog_path)
            self.assertGreater(len(entries_after), count_before)
            # New entry should have status=pending
            new_entries = [e for e in entries_after if e.get("status") == "pending"
                           and e not in entries_before]
            # Find entry with request_count=1 and last_seen=abc9999
            appended = next(
                (e for e in entries_after if e.get("last_seen") == "abc9999"
                 and e.get("request_count") == "1"),
                None
            )
            self.assertIsNotNone(appended)


def _make_crash_test_env(tmpdir: Path, mod, log_fixture: str, counter_start: int = 0):
    """Helper: set up a minimal environment for crash-trigger tests."""
    backlog_path = tmpdir / "backlog.md"
    backlog_path.write_text(BOOTSTRAP_BACKLOG, encoding="utf-8")
    counter_path = tmpdir / "crash-counter.txt"
    counter_path.write_text(str(counter_start) + "\n", encoding="utf-8")
    last_eval_log = tmpdir / ".omc" / "last_eval.log"
    last_eval_log.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / log_fixture, last_eval_log)

    overrides = {
        "_BACKLOG_PATH": backlog_path,
        "_RESEARCH_NOTES_PATH": tmpdir / "research_notes.md",
        "_RETEST_SENTINEL": tmpdir / ".omc" / "retest-in-progress",
        "_MAINTAIN_DISABLED_SENTINEL": tmpdir / ".omc" / "maintainer-disabled",
        "_CRASH_COUNTER_PATH": counter_path,
        "_SPECS_DIR": tmpdir / "specs",
        "REPO_ROOT": tmpdir,
    }
    originals = {k: getattr(mod, k) for k in overrides}
    (tmpdir / "research_notes.md").write_text("", encoding="utf-8")
    for k, v in overrides.items():
        setattr(mod, k, v)
    return originals, counter_path


import shutil


class TestCrashTriggerPipelineBugHalts(unittest.TestCase):
    """test_crash_trigger_pipeline_bug_halts: counter >= 3 on pipeline_bug -> exit 1."""

    def test_crash_trigger_pipeline_bug_halts(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            originals, counter_path = _make_crash_test_env(
                tmpdir, mod, "pipeline_bug.log", counter_start=2
            )
            try:
                rc = mod.run_maintain("crash")
            finally:
                for k, v in originals.items():
                    setattr(mod, k, v)
            self.assertEqual(rc, 1)  # halt
            self.assertEqual(int(counter_path.read_text().strip()), 3)


class TestCrashTriggerHypothesisContentContinues(unittest.TestCase):
    """test_crash_trigger_hypothesis_content_continues: no traceback, exit 0 -> continue."""

    def test_crash_trigger_hypothesis_content_continues(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            originals, counter_path = _make_crash_test_env(
                tmpdir, mod, "hypothesis_content.log", counter_start=1
            )
            try:
                rc = mod.run_maintain("crash")
            finally:
                for k, v in originals.items():
                    setattr(mod, k, v)
            self.assertEqual(rc, 0)  # continue
            # hypothesis_content resets counter to 0
            self.assertEqual(int(counter_path.read_text().strip()), 0)


class TestCrashTriggerRetrainCrashHalts(unittest.TestCase):
    """test_crash_trigger_retrain_crash_halts: retrain traceback -> counter increments, halt at 3."""

    def test_crash_trigger_retrain_crash_halts(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            originals, counter_path = _make_crash_test_env(
                tmpdir, mod, "retrain_crash.log", counter_start=2
            )
            try:
                rc = mod.run_maintain("crash")
            finally:
                for k, v in originals.items():
                    setattr(mod, k, v)
            self.assertEqual(rc, 1)  # halt
            self.assertEqual(int(counter_path.read_text().strip()), 3)


class TestCrashTriggerUnclassifiableContinues(unittest.TestCase):
    """test_crash_trigger_unclassifiable_continues: unknown frame -> exit 0, counter unchanged."""

    def test_crash_trigger_unclassifiable_continues(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            originals, counter_path = _make_crash_test_env(
                tmpdir, mod, "unclassifiable.log", counter_start=1
            )
            try:
                rc = mod.run_maintain("crash")
            finally:
                for k, v in originals.items():
                    setattr(mod, k, v)
            self.assertEqual(rc, 0)  # continue
            # Counter should NOT be incremented for unclassifiable
            self.assertEqual(int(counter_path.read_text().strip()), 1)


class TestPeriodicTriggerDraftsSpecForLowRiskPending(unittest.TestCase):
    """test_periodic_trigger_drafts_spec_for_low_risk_pending: eligible entry -> spec file created."""

    def test_periodic_trigger_drafts_spec_for_low_risk_pending(self):
        mod = _load_supervisor_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a backlog with a low-risk pending entry with request_count >= 3
            backlog_content = """\
# Enhancement Backlog

**Purpose:** Track requests.

## per-class-oof-f1

- **status:** pending
- **first_seen:** abc1234
- **last_seen:** abc5678
- **request_count:** 3
- **category:** observability
- **risk:** low (train_classifier.py emit expansion + prompt composer additional line)
- **excerpt:** Per-class OOF F1 in CURRENT STATE alongside the aggregate weighted F1.
- **notes:** Pure additive change; no risk.
"""
            backlog_path = tmpdir / "backlog.md"
            backlog_path.write_text(backlog_content, encoding="utf-8")

            notes_path = tmpdir / "research_notes.md"
            notes_path.write_text("", encoding="utf-8")

            specs_dir = tmpdir / "specs"
            omc_dir = tmpdir / ".omc"
            omc_dir.mkdir(parents=True, exist_ok=True)

            orig_backlog = mod._BACKLOG_PATH
            orig_notes = mod._RESEARCH_NOTES_PATH
            orig_retest = mod._RETEST_SENTINEL
            orig_disabled = mod._MAINTAIN_DISABLED_SENTINEL
            orig_counter = mod._CRASH_COUNTER_PATH
            orig_specs = mod._SPECS_DIR
            orig_root = mod.REPO_ROOT

            mod._BACKLOG_PATH = backlog_path
            mod._RESEARCH_NOTES_PATH = notes_path
            mod._RETEST_SENTINEL = tmpdir / ".omc" / "retest-in-progress"
            mod._MAINTAIN_DISABLED_SENTINEL = tmpdir / ".omc" / "maintainer-disabled"
            mod._CRASH_COUNTER_PATH = tmpdir / ".omc" / "crash-counter.txt"
            mod._SPECS_DIR = specs_dir
            mod.REPO_ROOT = tmpdir

            try:
                rc = mod.run_maintain("periodic")
            finally:
                mod._BACKLOG_PATH = orig_backlog
                mod._RESEARCH_NOTES_PATH = orig_notes
                mod._RETEST_SENTINEL = orig_retest
                mod._MAINTAIN_DISABLED_SENTINEL = orig_disabled
                mod._CRASH_COUNTER_PATH = orig_counter
                mod._SPECS_DIR = orig_specs
                mod.REPO_ROOT = orig_root

            self.assertEqual(rc, 0)
            spec_file = specs_dir / "deep-interview-per-class-oof-f1.md"
            self.assertTrue(spec_file.exists(),
                            f"Expected spec file at {spec_file}")
            # Entry status should be updated to spec_drafted
            _, entries = mod._parse_backlog(backlog_path)
            entry = next(e for e in entries if e.get("id") == "per-class-oof-f1")
            self.assertEqual(entry.get("status"), "spec_drafted")


if __name__ == "__main__":
    unittest.main()
