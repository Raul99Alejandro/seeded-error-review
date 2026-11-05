"""Run one reviewer over documents. Results are appended, so an interrupted run can resume."""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

from reviewers import SYSTEM, HaikuCliReviewer, MockReviewer, OpusCliReviewer, OpusReviewer, QwenReviewer, build_user_prompt

OPUS_INPUT_USD_PER_MTOK = 4.00
OPUS_OUTPUT_USD_PER_MTOK = 20.00
ESTIMATED_OUTPUT_TOKENS = 3000  # Assumption until dev runs give a measured number.


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_docs(docs, args):
    chosen = [d for d in docs if d["split"] == args.split and (args.size == "all" or d["size"] == args.size)]
    return chosen[: args.limit] if args.limit else chosen


def estimate_cost(docs, contexts, passes):
    """Rough estimate at about 4 characters per token. Ignores thinking tokens beyond the assumption."""
    chars = sum(len(SYSTEM) + len(build_user_prompt(d, c)) for d in docs for c in contexts)
    input_tokens = chars / 4 * passes
    calls = len(docs) * len(contexts) * passes
    output_tokens = calls * ESTIMATED_OUTPUT_TOKENS
    usd = input_tokens / 1e6 * OPUS_INPUT_USD_PER_MTOK + output_tokens / 1e6 * OPUS_OUTPUT_USD_PER_MTOK
    return calls, input_tokens, output_tokens, usd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", choices=["opus", "opus-cli", "haiku-cli", "qwen", "mock"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--size", choices=["small", "large", "all"], default="all")
    parser.add_argument("--context", choices=["fresh", "saturated", "both", "same_session"], default="both")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/results.jsonl"))
    parser.add_argument("--estimate", action="store_true", help="Print estimated cost and exit")
    parser.add_argument("--spend", action="store_true", help="Required to make paid Opus calls")
    parser.add_argument("--max-calls", type=int, default=0, help="Stop after this many new calls")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first call with status error")
    args = parser.parse_args()
    if args.context == "same_session" and args.reviewer != "opus-cli":
        raise SystemExit("--context same_session is only implemented for --reviewer opus-cli.")

    docs = select_docs(read_jsonl(args.data / "docs.jsonl"), args)
    contexts = ["fresh", "saturated"] if args.context == "both" else [args.context]

    calls, input_tokens, output_tokens, usd = estimate_cost(docs, contexts, args.passes)
    print(f"{calls} calls, about {input_tokens:,.0f} input and {output_tokens:,.0f} output tokens")
    if args.reviewer == "opus":
        print(f"Estimated Opus cost at standard rates: USD {usd:.2f}")
    if args.estimate:
        return
    if args.reviewer == "opus" and not args.spend:
        raise SystemExit("Opus calls cost money. Check the estimate and add --spend to run.")

    if args.reviewer == "opus":
        reviewer = OpusReviewer()
    elif args.reviewer == "opus-cli":
        reviewer = OpusCliReviewer()
    elif args.reviewer == "haiku-cli":
        reviewer = HaikuCliReviewer()
    elif args.reviewer == "qwen":
        reviewer = QwenReviewer()
    else:
        reviewer = MockReviewer(read_jsonl(args.data / "truth.jsonl"))

    # Call errors are retried on resume. Refusals and truncations stay as measured outcomes.
    done = {(r["doc_id"], r["reviewer"], r["context"], r["pass"]) for r in read_jsonl(args.out) if r["status"] != "error"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    made = 0
    with args.out.open("a", encoding="utf-8") as f:
        for doc, context, pass_index in product(docs, contexts, range(args.passes)):
            if (doc["doc_id"], reviewer.name, context, pass_index) in done:
                continue
            if args.max_calls and made >= args.max_calls:
                print(f"Stopped at --max-calls {args.max_calls}.")
                return
            result = reviewer.review(doc, context, pass_index)
            made += 1
            row = {"doc_id": doc["doc_id"], "reviewer": reviewer.name, "context": context, "pass": pass_index, **result}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{doc['doc_id']} {reviewer.name} {context} pass={pass_index} {result['status']} findings={len(result['findings'])}")
            if args.stop_on_error and result["status"] == "error":
                sys.exit(f"Stopped at the first error: {result['detail']}")


if __name__ == "__main__":
    main()
