"""Checks the pipeline without calling any model."""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run
from generate import SIZES, build_catalog, build_quote, build_terms, effective_discount, generate, iva_rate, percent, unit_price
from reviewers import REVIEW_TURN, SESSION_SYSTEM, SCHEMA, SYSTEM, HaikuCliReviewer, MockReviewer, OpusCliReviewer, build_user_prompt
from score import latest_rows, merge_passes, recall, recall_by_type, score_condition, score_doc

HERE = Path(__file__).parent


class GenerateTests(unittest.TestCase):
    def test_same_seed_gives_same_data(self):
        self.assertEqual(generate(1, 2, 2), generate(1, 2, 2))

    def test_clean_quote_applies_every_rule(self):
        rng = random.Random(3)
        catalog = build_catalog(rng, "large")
        terms = build_terms(rng)
        skus, f = build_quote(rng, catalog, "large", terms)
        base = 0
        for n, sku in enumerate(skus, start=1):
            self.assertEqual(f[f"line:{n}:unit_price"], unit_price(catalog[sku], f[f"line:{n}:qty"], terms["quote_date"]))
            self.assertEqual(f[f"line:{n}:line_total"], f[f"line:{n}:qty"] * f[f"line:{n}:unit_price"])
            if catalog[sku]["category"] != "SVC":
                base += f[f"line:{n}:line_total"]
        lines = SIZES["large"]["lines"]
        self.assertEqual(f["subtotal"], sum(f[f"line:{n}:line_total"] for n in range(1, lines + 1)))
        self.assertEqual(f["discount_amount"], percent(base, effective_discount(terms)))
        self.assertEqual(f["iva"], percent(f["subtotal"] - f["discount_amount"], iva_rate(terms)))
        self.assertEqual(f["total"], f["subtotal"] - f["discount_amount"] + f["iva"])

    def test_volume_price_takes_priority_over_dated_price(self):
        item = {"price": 1000, "tier": {"min_qty": 10, "price": 900}, "dated": {"until": "2026-08-01", "old_price": 950}}
        self.assertEqual(unit_price(item, 10, "2026-07-01"), 900)
        self.assertEqual(unit_price(item, 9, "2026-07-01"), 950)
        self.assertEqual(unit_price(item, 9, "2026-09-01"), 1000)

    def test_amendment_applies_from_its_effective_date(self):
        terms = {"contract_pct": 5, "amendment_pct": 10, "amendment_effective": "2026-08-01", "quote_date": "2026-08-01"}
        self.assertEqual(effective_discount(terms), 10)
        self.assertEqual(effective_discount({**terms, "quote_date": "2026-07-31"}), 5)

    def test_percent_rounds_half_up(self):
        self.assertEqual(percent(50, 1), 1)
        self.assertEqual(percent(49, 1), 0)

    def test_every_error_type_appears_and_rule_errors_dominate(self):
        _, truth = generate(2, 60, 0)
        kinds = [k for row in truth for k in row["seeded_types"].values()]
        expected = {"arithmetic", "tier_ignored", "tier_misapplied", "wrong_period_price", "discount_on_excluded", "superseded_discount", "wrong_iva_rate"}
        self.assertEqual(set(kinds), expected)
        self.assertGreater(sum(k != "arithmetic" for k in kinds), sum(k == "arithmetic" for k in kinds))

    def test_error_types_are_not_written_in_the_document(self):
        docs, truth = generate(2, 3, 3)
        for doc, row in zip(docs, truth):
            self.assertNotIn("seeded", json.dumps(doc))
            for kind in row["seeded_types"].values():
                self.assertNotIn(kind, doc["text"])

    def test_only_seeded_fields_differ_from_truth(self):
        _, truth = generate(2, 10, 10)
        for row in truth:
            changed = sorted(k for k in row["fields"] if row["fields"][k] != row["shown"][k])
            self.assertEqual(changed, row["seeded"])

    def test_document_does_not_leak_truth(self):
        docs, truth = generate(2, 3, 3)
        for doc in docs:
            self.assertNotIn("seeded", doc)
            self.assertNotIn("fields", doc)

    def test_only_saturated_prompt_has_the_log(self):
        docs, _ = generate(2, 1, 0)
        self.assertNotIn("author-agent", build_user_prompt(docs[0], "fresh"))
        self.assertIn("author-agent", build_user_prompt(docs[0], "saturated"))


class ScoreTests(unittest.TestCase):
    def test_recall_by_type(self):
        truth = {"d": {"seeded": ["a", "b"], "seeded_types": {"a": "x", "b": "y"}, "fields": {"a": 1, "b": 2}}}
        rows = [{"doc_id": "d", "pass": 0, "findings": [{"field": "a", "corrected_value": "0.01"}]}]
        self.assertEqual(recall_by_type(rows, truth), {"x": (1, 1), "y": (0, 1)})

    truth = {"fields": {"a": 100, "b": 200, "c": 300}, "shown": {"a": 150, "b": 200, "c": 300}, "seeded": ["a"]}

    def test_correct_fix(self):
        s = score_doc(self.truth, {"a": 100})
        self.assertEqual((s["caught"], s["correct_fixes"], s["wrong_fixes"], s["introduced"]), (1, 1, 0, 0))

    def test_wrong_fix_is_caught_but_not_correct(self):
        s = score_doc(self.truth, {"a": 120})
        self.assertEqual((s["caught"], s["correct_fixes"], s["wrong_fixes"]), (1, 0, 1))

    def test_changing_a_correct_field_introduces_an_error(self):
        s = score_doc(self.truth, {"b": 999})
        self.assertEqual((s["caught"], s["false_alarms"], s["introduced"]), (0, 1, 1))

    def test_flagging_a_correct_field_with_its_true_value_adds_no_error(self):
        s = score_doc(self.truth, {"b": 200})
        self.assertEqual((s["false_alarms"], s["introduced"]), (1, 0))

    def test_nonexistent_field_counts_as_new_error(self):
        s = score_doc(self.truth, {"nope": 1})
        self.assertEqual((s["false_alarms"], s["introduced"]), (1, 1))

    def test_merge_uses_union_and_majority(self):
        rows = [
            {"findings": [{"field": "a", "corrected_value": "1.00"}]},
            {"findings": [{"field": "a", "corrected_value": "1.00"}, {"field": "b", "corrected_value": "2.00"}]},
            {"findings": [{"field": "a", "corrected_value": "9.00"}]},
        ]
        self.assertEqual(merge_passes(rows), {"a": 100, "b": 200})

    def test_retry_replaces_the_failed_attempt(self):
        key = {"doc_id": "d", "reviewer": "r", "context": "fresh", "pass": 0}
        rows = latest_rows([{**key, "status": "error"}, {**key, "status": "ok"}])
        self.assertEqual([r["status"] for r in rows], ["ok"])

    def test_brackets_around_a_field_id_do_not_count_as_another_field(self):
        merged = merge_passes([{"findings": [{"field": "[a]", "corrected_value": "1.00"}]}])
        self.assertEqual(merged, {"a": 100})

    def test_unparseable_value_counts_as_wrong(self):
        merged = merge_passes([{"findings": [{"field": "a", "corrected_value": "n/a"}]}])
        self.assertEqual(score_doc(self.truth, merged)["wrong_fixes"], 1)

    def test_more_passes_never_lower_mock_recall(self):
        docs, truth = generate(5, 0, 30)
        truth_by_id = {t["doc_id"]: t for t in truth}
        mock = MockReviewer(truth, miss=0.5)
        rows = [
            {"doc_id": d["doc_id"], "pass": p, **mock.review(d, "fresh", p)}
            for d in docs for p in range(5)
        ]
        one = recall(list(score_condition(rows, truth_by_id, 1).values()))
        five = recall(list(score_condition(rows, truth_by_id, 5).values()))
        self.assertGreater(five, one)


class EndToEndTests(unittest.TestCase):
    def test_mock_run_resumes_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "r.jsonl"
            def call(script, *args):
                return subprocess.run([sys.executable, str(HERE / script), *args], check=True, capture_output=True, text=True).stdout
            call("generate.py", "--dev", "3", "--test", "0", "--out", str(data))
            run_args = ["--reviewer", "mock", "--passes", "2", "--data", str(data), "--out", str(out)]
            call("run.py", *run_args)
            first = out.read_text().count("\n")
            call("run.py", *run_args)
            self.assertEqual(first, out.read_text().count("\n"))
            self.assertEqual(first, 3 * 2 * 2 * 2)  # docs x sizes x contexts x passes
            report = call("score.py", "--data", str(data), "--results", str(out))
            self.assertIn("| mock | fresh | large | 2 | 3 |", report)

    def test_opus_refuses_to_spend_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            subprocess.run([sys.executable, str(HERE / "generate.py"), "--dev", "1", "--test", "0", "--out", str(data)], check=True, capture_output=True)
            result = subprocess.run(
                [sys.executable, str(HERE / "run.py"), "--reviewer", "opus", "--data", str(data), "--out", str(Path(tmp) / "r.jsonl")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--spend", result.stderr)


class FakeClaude:
    """Stands in for subprocess.run, so these tests never reach Claude Code."""

    def __init__(self, reply=None, returncode=0, stderr=b"", raises=None, init_overrides=None, extra_events=(), stdout=None):
        self.reply = {
            "is_error": False,
            "structured_output": {"findings": []},
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 2},
            "modelUsage": {"claude-opus-5-5": {}},
        }
        self.reply.update(reply or {})
        self.returncode, self.stderr, self.raises = returncode, stderr, raises
        self.init_overrides, self.extra_events, self.stdout = init_overrides or {}, extra_events, stdout
        self.calls = []

    def __call__(self, cmd, **kwargs):
        cwd = Path(kwargs["cwd"])
        self.calls.append({"cmd": cmd, "cwd_was_empty": cwd.is_dir() and not list(cwd.iterdir()), **kwargs})
        if cmd[1:3] == ["auth", "status"]:
            return subprocess.CompletedProcess(cmd, 0, b'{"loggedIn": true, "authMethod": "claude.ai", "apiProvider": "firstParty"}', b"")
        if self.raises:
            raise self.raises
        init = {
            "type": "system", "subtype": "init", "cwd": kwargs["cwd"], "apiKeySource": "none",
            "tools": ["StructuredOutput"], "mcp_servers": [], "model": "claude-opus-5-5", **self.init_overrides,
        }
        events = [init, *self.extra_events, {"type": "result", **self.reply}]
        out = self.stdout if self.stdout is not None else "\n".join(json.dumps(e) for e in events).encode()
        return subprocess.CompletedProcess(cmd, self.returncode, out, self.stderr)

    def review_calls(self):
        return [c for c in self.calls if c["cmd"][1] == "-p"]


class OpusCliTests(unittest.TestCase):
    DOC = {"doc_id": "d", "text": "QUOTE \u00f1", "work_log": "LOG"}

    def run_review(self, fake, context="fresh"):
        with mock.patch("reviewers.shutil.which", return_value="claude"), mock.patch("reviewers.subprocess.run", fake):
            return OpusCliReviewer().review(self.DOC, context, 0)

    def assert_error(self, result, starts_with):
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["findings"], [])
        self.assertTrue(result["detail"].startswith(starts_with), result["detail"])

    def test_api_credentials_and_session_state_do_not_reach_the_subprocess(self):
        fake = FakeClaude()
        planted = {"ANTHROPIC_API_KEY": "x", "ANTHROPIC_AUTH_TOKEN": "x", "ANTHROPIC_BASE_URL": "x", "CLAUDECODE": "1", "CLAUDE_CODE_SIMPLE": "1"}
        with mock.patch.dict(os.environ, {**planted, "KEEP_ME": "1"}):
            self.run_review(fake)
        self.assertEqual(len(fake.calls), 2)  # The auth check is covered as well.
        for call in fake.calls:
            for name in planted:
                self.assertNotIn(name, call["env"])
            self.assertEqual(call["env"]["KEEP_ME"], "1")

    def test_runs_in_new_empty_temp_directory_outside_project(self):
        fake = FakeClaude()
        self.run_review(fake)
        self.run_review(fake)
        calls = fake.review_calls()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["cwd"], calls[1]["cwd"])
        for call in calls:
            self.assertTrue(call["cwd_was_empty"])
            self.assertFalse(Path(call["cwd"]).resolve().is_relative_to(HERE.resolve()))
            self.assertFalse(Path(call["cwd"]).exists())

    def test_refuses_a_temp_directory_inside_the_project(self):
        fake = FakeClaude()
        with mock.patch("reviewers.PROJECT_DIR", Path(tempfile.gettempdir()).resolve()):
            self.assert_error(self.run_review(fake), "isolation")
        self.assertEqual(fake.review_calls(), [])

    def test_command_is_isolated_and_uses_the_shared_prompts(self):
        fake = FakeClaude()
        self.run_review(fake, "saturated")
        call = fake.review_calls()[0]
        cmd = call["cmd"]
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(cmd[cmd.index("--model") + 1], "claude-opus-5-5")
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], SYSTEM)
        self.assertEqual(json.loads(cmd[cmd.index("--json-schema") + 1]), SCHEMA)
        for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
            self.assertIn(flag, cmd)
        for flag in ("-c", "--continue", "-r", "--resume", "--session-id", "--add-dir", "--fallback-model"):
            self.assertNotIn(flag, cmd)
        self.assertEqual(call["input"].decode("utf-8"), build_user_prompt(self.DOC, "saturated"))

    def run_haiku(self, fake):
        with mock.patch("reviewers.shutil.which", return_value="claude"), mock.patch("reviewers.subprocess.run", fake):
            return HaikuCliReviewer().review(self.DOC, "fresh", 0)

    def test_haiku_requests_and_requires_its_own_model(self):
        haiku = {"modelUsage": {"claude-haiku-5-5": {}}}
        fake = FakeClaude(haiku, init_overrides={"model": "claude-haiku-5-5"})
        result = self.run_haiku(fake)
        self.assertEqual(result["status"], "ok")
        cmd = fake.review_calls()[0]["cmd"]
        self.assertEqual(cmd[cmd.index("--model") + 1], "claude-haiku-5-5")

    def test_haiku_rejects_a_run_that_used_opus(self):
        self.assert_error(self.run_haiku(FakeClaude()), "model")

    def test_same_session_continues_one_conversation_and_reviews_last(self):
        finding = {"field": "total", "problem": "p", "corrected_value": "1.00"}
        fake = FakeClaude({"result": "reply", "structured_output": {"findings": [finding]}})
        result = self.run_review(fake, "same_session")
        calls = fake.review_calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({c["cwd"] for c in calls}), 1)
        self.assertTrue(calls[0]["cwd_was_empty"])
        session_id = calls[0]["cmd"][calls[0]["cmd"].index("--session-id") + 1]
        for call in calls[1:]:
            self.assertEqual(call["cmd"][call["cmd"].index("--resume") + 1], session_id)
        for call, schema in zip(calls, (False, False, True)):
            cmd = call["cmd"]
            self.assertEqual("--json-schema" in cmd, schema)
            self.assertEqual(cmd[cmd.index("--tools") + 1], "")
            self.assertEqual(cmd[cmd.index("--system-prompt") + 1], SESSION_SYSTEM)
            self.assertIn("--safe-mode", cmd)
            self.assertNotIn("--no-session-persistence", cmd)
        self.assertIn("QUOTE ñ", calls[0]["input"].decode("utf-8"))
        self.assertEqual(calls[2]["input"].decode("utf-8"), REVIEW_TURN)
        self.assertEqual(result["findings"], [finding])
        self.assertEqual(result["author_replies"], ["reply", "reply"])

    def test_same_session_stops_at_a_failed_author_turn(self):
        fake = FakeClaude({"is_error": True, "result": "boom"})
        self.assert_error(self.run_review(fake, "same_session"), "cli")
        self.assertEqual(len(fake.review_calls()), 1)

    def test_same_session_keeps_isolation_checks(self):
        fake = FakeClaude(init_overrides={"tools": ["Read"]})
        self.assert_error(self.run_review(fake, "same_session"), "isolation")
        self.assertEqual(len(fake.review_calls()), 1)

    def test_valid_reply_is_recorded_with_tokens_and_duration(self):
        finding = {"field": "total", "problem": "p", "corrected_value": "1.00"}
        result = self.run_review(FakeClaude({"structured_output": {"findings": [finding]}}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["findings"], [finding])
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (15, 5))
        self.assertEqual(result["format_source"], "structured_output")
        self.assertEqual(result["model"], "claude-opus-5-5")
        self.assertIn("duration_s", result)

    def test_plain_json_reply_is_accepted_and_labelled(self):
        result = self.run_review(FakeClaude({"structured_output": None, "result": '{"findings": []}'}))
        self.assertEqual((result["status"], result["format_source"]), ("ok", "result_text"))

    def test_invalid_json_is_an_error(self):
        self.assert_error(self.run_review(FakeClaude({"structured_output": None, "result": "Looks fine!"})), "invalid_json")

    def test_reply_breaking_the_schema_is_an_error(self):
        for payload in (
            {"findings": [{"field": "total"}]}, {"findings": "none"}, {"findings": [], "extra": 1}, [],
            {"findings": [{"field": "t", "problem": "p", "corrected_value": 1.0}]},
        ):
            self.assert_error(self.run_review(FakeClaude({"structured_output": payload})), "invalid_json")

    def test_usage_limit_is_an_error(self):
        result = self.run_review(FakeClaude({"is_error": True, "result": "Claude usage limit reached", "usage": {}}, returncode=1))
        self.assert_error(result, "usage_limit")
        self.assertEqual((result["input_tokens"], result["output_tokens"], result["tokens_reported"]), (0, 0, False))

    def test_cli_failures_are_errors(self):
        self.assert_error(self.run_review(FakeClaude(returncode=1, stderr=b"boom", stdout=b"")), "cli")
        self.assert_error(self.run_review(FakeClaude(stdout=b"not json")), "cli")
        self.assert_error(self.run_review(FakeClaude({"is_error": True, "result": "bad"})), "cli")

    def test_timeout_and_missing_program_are_errors(self):
        self.assert_error(self.run_review(FakeClaude(raises=subprocess.TimeoutExpired("claude", 1))), "timeout")
        self.assert_error(self.run_review(FakeClaude(raises=FileNotFoundError("claude"))), "cli")

    def test_broken_isolation_is_an_error_even_with_good_findings(self):
        for overrides in ({"apiKeySource": "ANTHROPIC_API_KEY"}, {"tools": ["Read", "StructuredOutput"]}, {"mcp_servers": [{"name": "x"}]}, {"cwd": str(HERE)}):
            self.assert_error(self.run_review(FakeClaude(init_overrides=overrides)), "isolation")
        tool_use = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}
        self.assert_error(self.run_review(FakeClaude(extra_events=[tool_use])), "isolation")

    def test_another_model_is_an_error(self):
        self.assert_error(self.run_review(FakeClaude(init_overrides={"model": "claude-sonnet-5-5"})), "model")
        self.assert_error(self.run_review(FakeClaude({"modelUsage": {"claude-sonnet-5-5": {}}})), "model")

    def test_refuses_to_start_without_a_subscription_login(self):
        answer = subprocess.CompletedProcess([], 0, b'{"loggedIn": true, "authMethod": "api_key", "apiProvider": "firstParty"}', b"")
        with mock.patch("reviewers.shutil.which", return_value="claude"), mock.patch("reviewers.subprocess.run", return_value=answer):
            with self.assertRaises(SystemExit):
                OpusCliReviewer()

    def run_cli(self, fake, tmp, *extra):
        data = Path(tmp) / "data"
        if not data.exists():
            subprocess.run([sys.executable, str(HERE / "generate.py"), "--dev", "2", "--test", "0", "--out", str(data)], check=True, capture_output=True)
        out = Path(tmp) / "r.jsonl"
        argv = ["run.py", "--reviewer", "opus-cli", "--size", "large", "--data", str(data), "--out", str(out), *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch("reviewers.shutil.which", return_value="claude"), mock.patch("reviewers.subprocess.run", fake), mock.patch("builtins.print"):
            run.main()
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    def test_max_calls_then_resume_repeats_no_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeClaude()
            self.assertEqual(len(self.run_cli(fake, tmp, "--max-calls", "1")), 1)
            rows = self.run_cli(fake, tmp)
            self.assertEqual(len({(r["doc_id"], r["context"]) for r in rows}), len(rows))
            self.assertEqual(len(fake.review_calls()), len(rows))
            self.assertGreater(len(rows), 1)

    def test_stop_on_error_records_the_failure_and_makes_no_more_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeClaude({"is_error": True, "result": "Claude usage limit reached"}, returncode=1)
            with self.assertRaises(SystemExit):
                self.run_cli(fake, tmp, "--stop-on-error")
            rows = [json.loads(line) for line in (Path(tmp) / "r.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["status"] for r in rows], ["error"])
            self.assertEqual(len(fake.review_calls()), 1)


if __name__ == "__main__":
    unittest.main()
