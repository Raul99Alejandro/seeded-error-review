"""Reviewers under test. All of them receive the same prompts and return the same shape."""
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

SYSTEM = (
    "You review commercial quotes for errors. The price list and customer terms are authoritative. "
    "Check each unit price, line total, subtotal, discount, IVA and total. Report only fields whose "
    "shown value is wrong, using the field id in brackets, and give the correct value as a plain "
    "number with two decimals. If nothing is wrong, return an empty list."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "problem": {"type": "string"},
                    "corrected_value": {"type": "string"},
                },
                "required": ["field", "problem", "corrected_value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def build_user_prompt(doc, context):
    if context == "fresh":
        return doc["text"]
    return f"AUTHOR SESSION LOG\n{doc['work_log']}\n\nDOCUMENT TO REVIEW\n{doc['text']}"


class OpusReviewer:
    name = "opus-5-5"

    def __init__(self):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()

    def review(self, doc, context, pass_index):
        anthropic = self.anthropic
        try:
            # Server-side fallbacks stay off on purpose: a silent model switch would mix conditions.
            response = self.client.messages.create(
                model="claude-opus-5-5",
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": build_user_prompt(doc, context)}],
                output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except anthropic.RateLimitError as err:
            return failure(f"rate_limit: {err}")
        except anthropic.APIStatusError as err:
            return failure(f"api_status_{err.status_code}: {err}")
        except anthropic.APIConnectionError as err:
            return failure(f"connection: {err}")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        if response.stop_reason != "end_turn":
            return {"status": response.stop_reason, "findings": [], **usage}
        text = next(block.text for block in response.content if block.type == "text")
        return {"status": "ok", "findings": json.loads(text)["findings"], **usage}


class OpusCliReviewer:
    """Opus through Claude Code on a Claude subscription, one isolated process per document.

    A call with any sign of API billing, extra tools, MCP servers or another model is recorded
    as an error, because it would not measure the intended condition.
    """

    name = "opus-cli"
    model = "claude-opus-5-5"

    def __init__(self):
        self.executable = shutil.which("claude")
        if not self.executable:
            raise SystemExit("Claude Code was not found on PATH.")
        self._require_subscription()

    def _require_subscription(self):
        """Costs no usage. Refuses to start unless Claude Code is logged in with a subscription."""
        with tempfile.TemporaryDirectory(prefix="opus-cli-", ignore_cleanup_errors=True) as cwd:
            try:
                completed = subprocess.run(
                    [self.executable, "auth", "status", "--json"],
                    cwd=cwd, env=cli_env(), capture_output=True, timeout=60,
                )
                status = json.loads(completed.stdout)
            except (OSError, subprocess.TimeoutExpired, ValueError) as err:
                raise SystemExit(f"Could not read Claude Code auth status: {err}")
        if not status.get("loggedIn") or status.get("authMethod") != "claude.ai" or status.get("apiProvider") != "firstParty":
            raise SystemExit("Claude Code is not logged in with a Claude subscription. No calls were made.")

    def review(self, doc, context, pass_index):
        started = time.monotonic()
        if context == SAME_SESSION:
            result = self._same_session_review(doc)
        else:
            result = self._call(build_user_prompt(doc, context))
        return {**result, "duration_s": round(time.monotonic() - started, 1)}

    def _call(self, prompt):
        with tempfile.TemporaryDirectory(prefix="opus-cli-", ignore_cleanup_errors=True) as cwd:
            if Path(cwd).resolve().is_relative_to(PROJECT_DIR):
                return failure(f"isolation: temporary directory {cwd} is inside the project")
            return self._run(CLI_FLAGS, prompt, cwd)

    def _same_session_review(self, doc):
        """The agent works with the quote over two turns, then reviews it in that same conversation."""
        session_id = str(uuid.uuid4())
        author_prompts = [AUTHOR_EMAIL_TURN.format(document=doc["text"]), AUTHOR_DELIVERY_TURN]
        replies, author_output_tokens = [], 0
        with tempfile.TemporaryDirectory(prefix="opus-cli-", ignore_cleanup_errors=True) as cwd:
            if Path(cwd).resolve().is_relative_to(PROJECT_DIR):
                return failure(f"isolation: temporary directory {cwd} is inside the project")
            for turn, prompt in enumerate(author_prompts):
                session_args = ["--session-id", session_id] if turn == 0 else ["--resume", session_id]
                result = self._run(session_flags(session_args, schema=False), prompt, cwd, expect_findings=False)
                if result["status"] != "ok":
                    return {**result, "detail": f"{result['detail']} (author turn {turn + 1})"}
                replies.append(result["text"])
                author_output_tokens += result["output_tokens"]
            result = self._run(session_flags(["--resume", session_id], schema=True), REVIEW_TURN, cwd)
        return {**result, "author_replies": replies, "author_output_tokens": author_output_tokens}

    def _run(self, flags, prompt, cwd, expect_findings=True):
        try:
            # The prompt goes through stdin: large quotes do not fit in a Windows command line.
            completed = subprocess.run(
                [self.executable, *with_model(flags, self.model)],
                cwd=cwd, env=cli_env(), input=prompt.encode("utf-8"), capture_output=True,
                timeout=CLI_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return failure(f"timeout: no answer after {CLI_TIMEOUT_SECONDS} s")
        except OSError as err:
            return failure(f"cli: {err}")
        return parse_cli_output(completed.stdout, completed.stderr, completed.returncode, cwd, expect_findings, self.model)


class HaikuCliReviewer(OpusCliReviewer):
    """The same isolated subscription mode with a less capable model."""

    name = "haiku-cli"
    model = "claude-haiku-5-5"


class QwenReviewer:
    """Local model through Ollama. Set OLLAMA_URL to reach the Mac mini."""

    name = "qwen-local"

    def __init__(self):
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.model = os.environ.get("QWEN_MODEL")
        if not self.model:
            raise SystemExit("Set QWEN_MODEL to the exact model tag shown by `ollama list`.")

    def review(self, doc, context, pass_index):
        body = {
            "model": self.model,
            "stream": False,
            "format": SCHEMA,
            "options": {"seed": pass_index + 1},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(doc, context)},
            ],
        }
        request = urllib.request.Request(
            self.url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                data = json.loads(response.read())
            findings = json.loads(data["message"]["content"])["findings"]
        except (OSError, KeyError, ValueError) as err:
            return failure(f"ollama: {err}")
        return {
            "status": "ok",
            "findings": findings,
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }


class MockReviewer:
    """For testing the pipeline only. It reads truth, so it never measures a real model."""

    name = "mock"

    def __init__(self, truth, miss=0.3, false_alarm=0.3, bad_fix=0.2):
        self.truth = {row["doc_id"]: row for row in truth}
        self.miss, self.false_alarm, self.bad_fix = miss, false_alarm, bad_fix

    def review(self, doc, context, pass_index):
        row = self.truth[doc["doc_id"]]
        rng = random.Random(f"{doc['doc_id']}:{context}:{pass_index}")
        findings = []
        for key in row["seeded"]:
            if rng.random() >= self.miss:
                value = row["fields"][key] + (100 if rng.random() < self.bad_fix else 0)
                findings.append({"field": key, "problem": "mock", "corrected_value": f"{value / 100:.2f}"})
        if rng.random() < self.false_alarm:
            clean = [k for k in row["fields"] if k not in row["seeded"] and not k.endswith(":qty")]
            key = rng.choice(clean)
            findings.append({"field": key, "problem": "mock", "corrected_value": f"{(row['fields'][key] + 100) / 100:.2f}"})
        return {"status": "ok", "findings": findings, "input_tokens": 0, "output_tokens": 0}


def failure(detail):
    return {"status": "error", "detail": detail, "findings": [], "input_tokens": 0, "output_tokens": 0}


# Options confirmed against `claude --help` for Claude Code 2.1.318.
CLI_MODEL = "claude-opus-5-5"
CLI_EFFORT = "high"
CLI_TIMEOUT_SECONDS = 1800
CLI_FLAGS = [
    "-p",
    "--output-format", "stream-json",
    "--verbose",  # Gives the init event used to verify isolation.
    "--model", CLI_MODEL,
    "--effort", CLI_EFFORT,
    "--system-prompt", SYSTEM,
    "--json-schema", json.dumps(SCHEMA),
    "--tools", "",
    "--safe-mode",  # No CLAUDE.md, skills, plugins, hooks, MCP servers or custom agents.
    "--setting-sources", "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
]
SAME_SESSION = "same_session"
SESSION_SYSTEM = "You are an account operations agent at a company that sells goods and services. Complete each request."
AUTHOR_EMAIL_TURN = (
    "You prepared the quote below earlier today. Write the cover email to the customer. "
    "State the total and mention the five largest lines.\n\n{document}"
)
AUTHOR_DELIVERY_TURN = (
    "Sales needs an internal delivery list for this quote: each SKU, its quantity and its line total. "
    "Then tell sales whether the quote is ready to send."
)
REVIEW_TURN = f"New task. {SYSTEM} Review the quote above exactly as it was shown."
STRIPPED_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE")  # Covers API credentials and parent-session state.
ALLOWED_CLI_TOOLS = {"StructuredOutput"}  # Added by --json-schema itself to return the answer.
PROJECT_DIR = Path(__file__).resolve().parent


def session_flags(session_args, schema):
    """Same isolation as CLI_FLAGS, except that the conversation persists across turns."""
    flags = [
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--model", CLI_MODEL,
        "--effort", CLI_EFFORT,
        "--system-prompt", SESSION_SYSTEM,
        "--tools", "",
        "--safe-mode",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        *session_args,
    ]
    return [*flags, "--json-schema", json.dumps(SCHEMA)] if schema else flags


def with_model(flags, model):
    flags = list(flags)
    flags[flags.index("--model") + 1] = model
    return flags


def cli_env():
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(STRIPPED_ENV_PREFIXES)}


def validate_findings(payload):
    """Check a reply against SCHEMA. Raises ValueError, so nothing malformed is ever scored."""
    if not isinstance(payload, dict) or set(payload) != set(SCHEMA["required"]):
        raise ValueError("top level must be an object with only 'findings'")
    if not isinstance(payload["findings"], list):
        raise ValueError("'findings' must be a list")
    keys = set(SCHEMA["properties"]["findings"]["items"]["required"])
    for finding in payload["findings"]:
        if not isinstance(finding, dict) or set(finding) != keys or not all(isinstance(v, str) for v in finding.values()):
            raise ValueError(f"finding does not match the schema: {str(finding)[:200]}")
    return payload["findings"]


def isolation_problem(events, init, cwd):
    if init is None:
        return "no init event, so isolation cannot be verified"
    if Path(str(init.get("cwd"))).resolve() != Path(cwd).resolve():
        return f"ran in {init.get('cwd')}, expected the temporary directory"
    if init.get("apiKeySource") not in (None, "none"):
        return f"API credential in use: {init['apiKeySource']}"
    extra_tools = set(init.get("tools") or []) - ALLOWED_CLI_TOOLS
    if extra_tools:
        return f"tools available: {sorted(extra_tools)}"
    if init.get("mcp_servers"):
        return f"MCP servers available: {init['mcp_servers']}"
    used = {
        block.get("name")
        for event in events if event.get("type") == "assistant"
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    } - ALLOWED_CLI_TOOLS
    if used:
        return f"tools used: {sorted(used)}"
    return None


def parse_cli_output(stdout, stderr, returncode, cwd, expect_findings=True, expected_model=CLI_MODEL):
    events = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    final = next((e for e in reversed(events) if e.get("type") == "result"), None)

    usage = (final or {}).get("usage") or {}
    cache_write = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    models_used = sorted((final or {}).get("modelUsage") or {})
    model = (init or {}).get("model")
    info = {
        # input_tokens counts cached tokens too, so it is comparable with a run without caching.
        "input_tokens": usage.get("input_tokens", 0) + cache_write + cache_read,
        "output_tokens": usage.get("output_tokens", 0),
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "tokens_reported": bool(usage),
        "model": model,
        "models_used": models_used,
        "api_key_source": (init or {}).get("apiKeySource"),
        "tools_available": (init or {}).get("tools"),
        "cli_version": (init or {}).get("claude_code_version"),
    }

    def error(detail):
        return {**failure(detail), **info}

    if returncode != 0 or final is None or final.get("is_error"):
        message = str((final or {}).get("result") or (final or {}).get("errors") or stderr.decode("utf-8", errors="replace"))[:500]
        limited = (final or {}).get("api_error_status") == 429 or any(
            word in message.lower() for word in ("usage limit", "rate limit", "limit reached", "hit your limit")
        )
        return error(f"{'usage_limit' if limited else 'cli'}: exit={returncode} {message}")

    problem = isolation_problem(events, init, cwd)
    if problem:
        return error(f"isolation: {problem}")
    if not str(model).startswith(expected_model) or not any(m.startswith(expected_model) for m in models_used):
        return error(f"model: expected {expected_model}, init reported {model}, usage reported {models_used}")

    if not expect_findings:
        return {"status": "ok", "findings": [], "text": str(final.get("result") or ""), **info}

    try:
        if final.get("structured_output") is not None:
            source, payload = "structured_output", final["structured_output"]
        else:
            source, payload = "result_text", json.loads(final.get("result") or "")
        findings = validate_findings(payload)
    except ValueError as err:
        return error(f"invalid_json: {err}")
    return {"status": "ok", "findings": findings, "format_source": source, **info}
