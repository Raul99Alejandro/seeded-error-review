"""Score results against truth. Bootstrap intervals resample documents, the unit of the experiment."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_cents(text):
    try:
        return round(float(str(text).replace(",", "").replace("$", "")) * 100)
    except ValueError:
        return None


def clean_field(field):
    """Documents show ids in brackets, and some models copy the brackets into their answer."""
    return field.strip().strip("[]")


def merge_passes(rows):
    """Union of findings across passes. If passes disagree on a value, the most common one wins."""
    votes = defaultdict(Counter)
    for row in rows:
        for finding in row["findings"]:
            votes[clean_field(finding["field"])][to_cents(finding["corrected_value"])] += 1
    return {field: counter.most_common(1)[0][0] for field, counter in votes.items()}


def score_doc(truth, merged):
    seeded = set(truth["seeded"])
    caught = correct_fixes = wrong_fixes = false_alarms = introduced = 0
    for field, value in merged.items():
        if field in seeded:
            caught += 1
            if value == truth["fields"][field]:
                correct_fixes += 1
            else:
                wrong_fixes += 1
        else:
            false_alarms += 1
            # Applying this correction would turn a correct field into a wrong one.
            if field not in truth["fields"] or value != truth["fields"][field]:
                introduced += 1
    return {
        "seeded": len(seeded), "caught": caught, "correct_fixes": correct_fixes,
        "wrong_fixes": wrong_fixes, "false_alarms": false_alarms, "introduced": introduced,
    }


def score_condition(rows, truth_by_id, k):
    """Score the first k passes per document. Failed passes contribute no findings."""
    by_doc = defaultdict(list)
    for row in rows:
        if row["pass"] < k:
            by_doc[row["doc_id"]].append(row)
    return {
        doc_id: {
            **score_doc(truth_by_id[doc_id], merge_passes(doc_rows)),
            "failed": sum(r["status"] != "ok" for r in doc_rows),
            "input_tokens": sum(r["input_tokens"] for r in doc_rows),
            "output_tokens": sum(r["output_tokens"] for r in doc_rows),
        }
        for doc_id, doc_rows in by_doc.items()
    }


def recall(scores):
    seeded = sum(s["seeded"] for s in scores)
    return sum(s["caught"] for s in scores) / seeded if seeded else 0.0


def recall_by_type(rows, truth_by_id):
    """Caught and seeded counts per error type, using the first pass of each document."""
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["pass"] != 0:
            continue
        truth = truth_by_id[row["doc_id"]]
        flagged = {clean_field(finding["field"]) for finding in row["findings"]}
        for field, kind in truth.get("seeded_types", {}).items():
            counts[kind][0] += field in flagged
            counts[kind][1] += 1
    return {kind: tuple(value) for kind, value in counts.items()}


def per_doc(scores, key):
    return sum(s[key] for s in scores) / len(scores) if scores else 0.0


def bootstrap_ci(values_fn, doc_ids, n=2000, seed=7):
    rng = random.Random(seed)
    samples = sorted(values_fn(rng.choices(doc_ids, k=len(doc_ids))) for _ in range(n))
    return samples[int(0.025 * n)], samples[int(0.975 * n) - 1]


def compare(a, b, metric):
    """Paired difference a - b over documents both conditions reviewed."""
    doc_ids = sorted(set(a) & set(b))
    if not doc_ids:
        return None
    def diff(ids):
        return metric([a[i] for i in ids]) - metric([b[i] for i in ids])
    return diff(doc_ids), *bootstrap_ci(diff, doc_ids), len(doc_ids)


def latest_rows(results):
    """Keep the last attempt of each call, so a retried error is not counted twice."""
    latest = {(r["doc_id"], r["reviewer"], r["context"], r["pass"]): r for r in results}
    return list(latest.values())


def conditions(results, docs_by_id, split):
    grouped = defaultdict(list)
    for row in latest_rows(results):
        doc = docs_by_id[row["doc_id"]]
        if doc["split"] == split:
            grouped[(row["reviewer"], row["context"], doc["size"])].append(row)
    return grouped


def hypothesis_rows(scored, reviewers, max_k):
    strong, weak = reviewers
    rows = []
    for size in ("large", "small"):
        a = scored.get((strong, "fresh", size, 1))
        b = scored.get((weak, "fresh", size, max_k.get((weak, "fresh", size), 1)))
        if a and b:
            rows.append((f"{size}: {strong} fresh vs {weak} all passes, recall", compare(a, b, recall)))
            rows.append((f"{size}: {strong} fresh vs {weak} all passes, new errors/doc", compare(a, b, lambda s: per_doc(s, "introduced"))))
        for model in reviewers:
            fresh = scored.get((model, "fresh", size, 1))
            for context in ("saturated", "same_session"):
                other = scored.get((model, context, size, 1))
                if fresh and other:
                    rows.append((f"{size}: {model} fresh vs {context}, recall", compare(fresh, other, recall)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results/results.jsonl"))
    parser.add_argument("--strong", default="opus-5-5")
    parser.add_argument("--weak", default="qwen-local")
    args = parser.parse_args()

    docs_by_id = {d["doc_id"]: d for d in read_jsonl(args.data / "docs.jsonl")}
    truth_by_id = {t["doc_id"]: t for t in read_jsonl(args.data / "truth.jsonl")}
    grouped = conditions(read_jsonl(args.results), docs_by_id, args.split)

    scored, max_k = {}, {}
    print("| reviewer | context | size | passes | docs | recall (95% CI) | wrong fixes/doc | false alarms/doc | new errors/doc | failed | input tok/doc | output tok/doc |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (reviewer, context, size), rows in sorted(grouped.items()):
        available = max(r["pass"] for r in rows) + 1
        max_k[(reviewer, context, size)] = available
        for k in sorted({1, 3, 5, available} & set(range(1, available + 1))):
            result = score_condition(rows, truth_by_id, k)
            scored[(reviewer, context, size, k)] = result
            ids = sorted(result)
            low, high = bootstrap_ci(lambda sample: recall([result[i] for i in sample]), ids)
            values = list(result.values())
            print(
                f"| {reviewer} | {context} | {size} | {k} | {len(ids)} | {recall(values):.3f} ({low:.3f}-{high:.3f}) "
                f"| {per_doc(values, 'wrong_fixes'):.2f} | {per_doc(values, 'false_alarms'):.2f} | {per_doc(values, 'introduced'):.2f} "
                f"| {sum(v['failed'] for v in values)} | {per_doc(values, 'input_tokens'):.0f} | {per_doc(values, 'output_tokens'):.0f} |"
            )

    by_type = {cond: recall_by_type(rows, truth_by_id) for cond, rows in sorted(grouped.items())}
    if any(by_type.values()):
        print("\n| reviewer | context | size | error type | caught / seeded |")
        print("| --- | --- | --- | --- | --- |")
        for (reviewer, context, size), counts in by_type.items():
            for kind, (caught, seeded) in sorted(counts.items()):
                print(f"| {reviewer} | {context} | {size} | {kind} | {caught} / {seeded} |")

    rows = hypothesis_rows(scored, (args.strong, args.weak), max_k)
    if rows:
        print("\n| comparison | difference (95% CI) | docs |")
        print("| --- | --- | --- |")
        for label, result in rows:
            if result:
                diff, low, high, n = result
                print(f"| {label} | {diff:+.3f} ({low:+.3f} to {high:+.3f}) | {n} |")


if __name__ == "__main__":
    main()
