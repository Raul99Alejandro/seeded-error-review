# Fresh eyes or many passes? Measuring AI reviewers on seeded errors

Status: the one pre-specified test comparison has been run. It did not support the claim that Haiku 5.5 introduces more new errors than Opus 5.5. The difference was -0.017 new errors per document (Haiku minus Opus; 95% paired bootstrap CI -0.250 to +0.200; 60 documents). That is not evidence the models are equal: each model's new errors came from a single document, so the interval is wide.

## Test result

Large test documents, fresh context, 1 pass each, 88 seeded errors, 17 clean documents. File hashes matched the frozen values before scoring, and all 120 calls succeeded.

| | Opus 5.5 | Haiku 5.5 |
| --- | --- | --- |
| Seeded errors caught | 88 / 88 | 88 / 88 |
| Wrong fixes | 0 | 0 |
| False alarms (all also new errors) | 5, in 1 document | 4, in 1 document |
| Clean documents flagged | 0 / 17 | 1 / 17 |
| Output tokens per document | 4,367 | 10,354 |

Descriptive notes on the two affected documents, checked by recomputing their totals:

- `large-test-021`, Opus: found both seeded errors, but computed 39 x 2400.94 as 93635.66 instead of 93636.66. The slip carried into the subtotal, discount, IVA and total.
- `large-test-048`, Haiku: a clean document. It summed the lines $1.00 too low and calculated the wrong goods base for the discount, which carried into IVA and the total.

What this does and does not show:

- Both models hit the recall ceiling, so these documents cannot separate them on catching errors.
- The dev lead did not replicate. On dev, a single Haiku pass introduced 0.27 new errors per document against 0.00 for Opus; on test it was 0.07 against 0.08. Dev was only 15 documents, and this question was formed from them.
- Hypotheses 1, 2 and 4 name Qwen, which was not run, so they remain untested. Hypothesis 3 (context) was not run on test, and on dev Opus was at the ceiling, so it remains unmeasured.
- Rare errors like these need many more documents to estimate. `arithmetic`, `discount_on_excluded` and `superseded_discount` had 4 seeded errors between them on test.

Next step, if continued: a real multi-part project with planted connected errors and intentional decisions that look like errors, with enough cases to estimate rare mistakes. It would compare the same model reviewing fresh, before compaction and after compaction, then vary the model and number of passes.

## Dev results (used to form the question, not evidence)

On the 15 large dev documents, Opus (subscription mode) caught all 24 seeded errors in the fresh, saturated and same-session conditions, with no false alarms or wrong fixes. `superseded_discount` errors did not appear in that sample.

Qwen has not been run. Haiku 5.5 (`--reviewer haiku-cli`) was run instead as the less capable model, so it tests the capability question, not the author's experience with Qwen. Its dev results: 119 of 120 seeded errors caught across 5 fresh passes, 12 false alarms against 0 for Opus's single pass, and 0.40 new errors per document after merging 5 passes (Opus 0.00; paired difference -0.40, 95% CI -1.07 to 0.00).

Scoring the test run:

```bash
python score.py --results results/final.jsonl --split test --strong opus-cli --weak haiku-cli
```

`score.py` prints Opus minus Haiku, so the sign above is flipped to match the pre-specified measure.

Claude wrote the code and ran the tests under the author's direction. The question and predictions come from the author's experience. Data is fully synthetic.

## Question

What makes an AI agent effective at catching errors in another agent's work: model capability, freedom from accumulated context, or number of passes?

## Hypotheses (fix these before running the test split)

On large documents:

1. Opus without context catches more seeded errors than Qwen.
2. Opus without context also introduces more new errors than Qwen.
3. For the same model, fresh reviewers catch more errors than reviewers saturated with context.
4. Qwen catches more errors as passes increase.

A hypothesis is supported only if the 95% paired bootstrap interval of the difference excludes zero. Otherwise it is not supported, and that gets reported too.

### Added after the dev runs (fixed on 2026-11-04, before any test document was reviewed)

This question came from the dev data, so it is tested on data not used to form it. It reverses the direction of hypothesis 2 and does not replace it.

- **Question:** does Haiku 5.5 introduce more new errors than Opus 5.5 when reviewing without context?
- **Data:** the 60 large test documents (88 seeded errors, 17 clean documents), run once.
- **Conditions:** `opus-cli` and `haiku-cli`, fresh context, 1 pass each, with default effort settings already in the code.
- **Primary measure:** new errors per document, Haiku minus Opus, 95% paired bootstrap interval over documents. It is supported only if the interval excludes zero and Haiku is higher. If the interval includes zero, it is not supported. If Opus is significantly higher, the result favors hypothesis 2.
- **Secondary, descriptive only:** recall, false alarms, wrong fixes, tokens and failed calls.
- **Frozen files (first 16 hex digits of SHA-256):** `data/docs.jsonl` c19cc97aae903c08, `data/truth.jsonl` 61ea163a734f55ae, `generate.py` a0a1211c50742b28, `reviewers.py` 5da62e72d5dc766b, `score.py` e3c1fbc4129fb132, `run.py` de74c7e700c08cfa.

The question, data and hashes above were written here before the test run, in a local working copy. This repository was published after the run, so its history cannot independently prove that timing.

Prompts, data and scoring do not change after this point. Any later analysis gets labeled as exploratory. Calls that fail are retried only for operational errors, and refusals stay as outcomes.

## Design

| Factor | Levels |
| --- | --- |
| Model | Claude Opus 5.5, Qwen (local through Ollama) |
| Context | Fresh (document only), saturated (author session log first) |
| Document size | Small (6 lines), large (60 lines) |
| Passes | 1, 3 and 5 for Qwen; Opus as budget allows |

Each synthetic quote has 0 to 3 seeded errors, and about one in four is clean. The saturated log adds noise and claims the quote was verified.

Correct values depend on rules in the document: volume prices, dated prices, service lines excluded from the discount, a dated contract amendment and IVA by delivery city. Error types are `tier_ignored`, `tier_misapplied`, `wrong_period_price`, `discount_on_excluded`, `superseded_discount`, `wrong_iva_rate` and `arithmetic`.

Metrics per condition: recall overall and by error type, wrong fixes, false alarms, new errors introduced, failed calls and tokens. New errors are corrections that would turn a correct field into a wrong one.

## Data versions

- `data-v1` and `results/pilot-subscription.jsonl`: arithmetic and lookup errors only. Opus caught all 5 seeded errors in a 3-document flow check, so these errors were replaced to leave room to measure a difference. Score with `--data data-v1`.
- `data`: the rule-based version above. The change was made before the test split was run, and the hypotheses did not change.

Rules:

- Reviewers never see the truth file.
- All reviewers receive the same prompts.
- Prompts may change only using the dev split; the test split runs once.
- Failed calls are counted, never dropped.

## Known limits

- Quotes rely on business rules, arithmetic and lookups, so results may not transfer to code or security review.
- The saturated condition is one specific choice of context: a log that also asserts the quote is correct. It mixes volume with misleading claims.
- The same-session condition (`--context same_session`, `opus-cli` only) tells the agent it prepared the quote, asks for a cover email and a delivery list, then asks for a review in that conversation. The errors are not really its own, and in the dev run its first reply already contained the correct value of all 24 seeded fields, usually in an unprompted check. That text match is a rough signal, but it suggests the review turn was not testing a blind spot. Its system prompt also differs from the fresh condition's, so any future difference needs a control with the same wrapper.
- Seeded errors are not adversarial, and no reviewer here is being deceived on purpose.
- Union across passes can raise false alarms along with recall, so read both.
- In `opus-cli` mode, Claude Code adds its own system prompt material around ours, so conditions are not identical to Qwen's, which receives only `SYSTEM` and the document. `--system-prompt` replaces the default prompt, but the exact text sent cannot be inspected from the CLI output.
- In `opus-cli` mode, the effective model and effort depend on the CLI. We request `claude-opus-5-5` with effort `high` and record the model the CLI reports, but the CLI does not report the effort applied. `--json-schema` also adds an internal `StructuredOutput` tool, the only tool available to the reviewer.

## Run

```bash
pip install -r requirements.txt
python -m unittest test_pipeline
python generate.py
python run.py --reviewer mock --passes 5
python score.py
```

Real runs, starting on the dev split. Use a separate results file so mock rows are never mixed in:

```bash
python run.py --reviewer opus --size large --limit 3 --out results/real.jsonl --estimate
python run.py --reviewer opus --size large --limit 3 --out results/real.jsonl --spend
```

```bash
OLLAMA_URL=http://<mac-mini>:11434 QWEN_MODEL=<tag> python run.py --reviewer qwen --passes 5 --out results/real.jsonl
```

### Claude subscription (no API key)

`opus-cli` runs Opus through Claude Code. It uses subscription usage instead of API billing, so it needs `claude auth login` with a subscription and no `--spend`:

```bash
python run.py --reviewer opus-cli --split dev --size large --limit 3 --context both --passes 1 --out results/pilot-subscription.jsonl --stop-on-error
python score.py --results results/pilot-subscription.jsonl --split dev --strong opus-cli
```

Add `--max-calls N` to cap new calls. Rerunning skips finished calls but retries calls with `status: error`.

Each call is a new `claude -p` process in a new, empty temporary directory outside the project. It has no tools, CLAUDE.md, MCP servers, settings files or session persistence, and every `ANTHROPIC_*` and `CLAUDE*` environment variable is removed. The run refuses to start without a subscription login. A call is recorded as `status: error`, with no findings, if the CLI fails, the usage limit is hit, the JSON is invalid, or its output shows an API credential, extra tools or another model.
