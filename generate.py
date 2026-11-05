"""Generate synthetic quotes with seeded errors. No real company data is used.

Money is stored as integer cents. Correct values depend on rules a reviewer must
apply: volume tiers, dated prices, discount exclusions, contract amendments and
delivery-based IVA. Each seeded error changes exactly one field, and truth is
written to a separate file that reviewers never see.
"""
import argparse
import datetime
import json
import random
from pathlib import Path

SIZES = {"small": {"lines": 6, "catalog": 15, "log": 40}, "large": {"lines": 60, "catalog": 150, "log": 400}}
SUMMARY_FIELDS = ["subtotal", "discount_amount", "iva", "total"]
BORDER_CITIES = ["Tijuana", "Ciudad Juarez", "Reynosa"]
OTHER_CITIES = ["Zacatecas", "Monterrey", "Guadalajara", "Puebla"]
STANDARD_IVA_PCT = 16
BORDER_IVA_PCT = 8
DISCOUNT_RATES = [5, 8, 10, 12, 15]


def money(cents):
    return f"{cents / 100:.2f}"


def percent(amount, pct):
    """Percentage of an amount in cents, rounded half up."""
    return (amount * pct + 50) // 100


def random_date(rng, start, end):
    days = (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days
    return (datetime.date.fromisoformat(start) + datetime.timedelta(days=rng.randint(0, days))).isoformat()


def build_catalog(rng, size):
    items = {}
    for i in range(SIZES[size]["catalog"]):
        price = rng.randint(500, 250000)
        item = {"category": "SVC" if rng.random() < 0.25 else "HW", "price": price}
        rule = rng.random()
        if rule < 0.25:
            item["tier"] = {"min_qty": rng.choice([10, 15, 20, 25]), "price": round(price * rng.choice([0.85, 0.9, 0.95]))}
        elif rule < 0.45:
            item["dated"] = {"until": random_date(rng, "2026-07-15", "2026-10-15"), "old_price": round(price * rng.choice([0.9, 0.95, 1.05]))}
        items[f"SKU-{i:04d}"] = item
    return items


def build_terms(rng):
    contract_pct, amendment_pct = rng.sample(DISCOUNT_RATES, 2)
    return {
        "quote_date": random_date(rng, "2026-07-01", "2026-10-31"),
        "city": rng.choice(BORDER_CITIES + OTHER_CITIES),
        "contract_pct": contract_pct,
        "amendment_pct": amendment_pct,
        "amendment_effective": random_date(rng, "2026-07-15", "2026-10-15"),
    }


def effective_discount(terms):
    return terms["amendment_pct"] if terms["quote_date"] >= terms["amendment_effective"] else terms["contract_pct"]


def iva_rate(terms):
    return BORDER_IVA_PCT if terms["city"] in BORDER_CITIES else STANDARD_IVA_PCT


def unit_price(item, qty, quote_date):
    if "tier" in item and qty >= item["tier"]["min_qty"]:
        return item["tier"]["price"]
    if "dated" in item and quote_date <= item["dated"]["until"]:
        return item["dated"]["old_price"]
    return item["price"]


def compute_summary(subtotal, discount_base, discount_pct, iva_pct):
    discount = percent(discount_base, discount_pct)
    iva = percent(subtotal - discount, iva_pct)
    return {"subtotal": subtotal, "discount_amount": discount, "iva": iva, "total": subtotal - discount + iva}


def build_quote(rng, catalog, size, terms):
    skus = rng.sample(sorted(catalog), SIZES[size]["lines"])
    # Every quote has a service line, so the discount exclusion always matters.
    if not any(catalog[sku]["category"] == "SVC" for sku in skus):
        catalog[skus[0]]["category"] = "SVC"
    fields = {}
    subtotal = discount_base = 0
    for n, sku in enumerate(skus, start=1):
        qty = rng.randint(1, 40)
        price = unit_price(catalog[sku], qty, terms["quote_date"])
        fields[f"line:{n}:qty"] = qty
        fields[f"line:{n}:unit_price"] = price
        fields[f"line:{n}:line_total"] = qty * price
        subtotal += qty * price
        if catalog[sku]["category"] != "SVC":
            discount_base += qty * price
    fields.update(compute_summary(subtotal, discount_base, effective_discount(terms), iva_rate(terms)))
    return skus, fields


def corrupt(rng, value):
    """Return a plausible wrong value: a small shift, a large shift or a slip."""
    kind = rng.choice(["small", "large", "slip"])
    if kind == "small":
        wrong = round(value * rng.choice([0.97, 0.98, 1.02, 1.03]))
    elif kind == "large":
        wrong = round(value * rng.choice([0.5, 0.7, 1.3, 1.5]))
    else:
        wrong = value + rng.choice([-1000, -100, 100, 1000])
    return wrong if wrong != value and wrong > 0 else value + 100


def rule_errors(catalog, skus, terms, fields):
    """Wrong values a person or agent would plausibly get by misapplying one rule."""
    errors = []
    for n, sku in enumerate(skus, start=1):
        item, key, qty = catalog[sku], f"line:{n}:unit_price", fields[f"line:{n}:qty"]
        if "tier" in item:
            if qty >= item["tier"]["min_qty"]:
                errors.append((key, item["price"], "tier_ignored"))
            else:
                errors.append((key, item["tier"]["price"], "tier_misapplied"))
        if "dated" in item:
            other = item["price"] if terms["quote_date"] <= item["dated"]["until"] else item["dated"]["old_price"]
            errors.append((key, other, "wrong_period_price"))
    subtotal, discount = fields["subtotal"], fields["discount_amount"]
    pct = effective_discount(terms)
    errors.append(("discount_amount", percent(subtotal, pct), "discount_on_excluded"))
    other_pct = terms["contract_pct"] if pct == terms["amendment_pct"] else terms["amendment_pct"]
    # Recover the discount base from the catalog so the superseded rate uses the same base.
    base = sum(fields[f"line:{n}:line_total"] for n, sku in enumerate(skus, start=1) if catalog[sku]["category"] != "SVC")
    errors.append(("discount_amount", percent(base, other_pct), "superseded_discount"))
    other_iva = STANDARD_IVA_PCT if iva_rate(terms) == BORDER_IVA_PCT else BORDER_IVA_PCT
    errors.append(("iva", percent(subtotal - discount, other_iva), "wrong_iva_rate"))
    return [e for e in errors if e[1] != fields[e[0]]]


def seed_errors(rng, catalog, skus, terms, fields, n_errors):
    arithmetic_keys = [k for k in fields if k.endswith(":line_total")] + ["subtotal", "total"]
    arithmetic = [(k, corrupt(rng, fields[k]), "arithmetic") for k in rng.sample(arithmetic_keys, 3)]
    candidates = rule_errors(catalog, skus, terms, fields) + arithmetic
    rng.shuffle(candidates)
    shown, types = dict(fields), {}
    for key, wrong, kind in candidates:
        if len(types) == n_errors:
            break
        if key not in types:
            shown[key] = wrong
            types[key] = kind
    return shown, types


def render(doc_id, catalog, skus, terms, shown):
    out = [
        f"QUOTE {doc_id}",
        f"Quote date: {terms['quote_date']}",
        f"Delivery city: {terms['city']}",
        "",
        "PRICE LIST (authoritative, MXN). Volume prices apply to the whole line and take priority over dated prices.",
    ]
    for sku, item in sorted(catalog.items()):
        row = f"{sku} | {item['category']} | {money(item['price'])}"
        if "tier" in item:
            row += f" | qty >= {item['tier']['min_qty']}: {money(item['tier']['price'])}"
        if "dated" in item:
            row += f" | through {item['dated']['until']}: {money(item['dated']['old_price'])}"
        out.append(row)
    out += [
        "",
        "CUSTOMER TERMS (authoritative)",
        f"Contract discount: {terms['contract_pct']}% on goods. Services (category SVC) are not discountable.",
        f"Amendment 1: for quotes dated on or after {terms['amendment_effective']}, the discount is {terms['amendment_pct']}%.",
        f"IVA: {STANDARD_IVA_PCT}% on subtotal minus discount, or {BORDER_IVA_PCT}% for delivery to {', '.join(BORDER_CITIES)}.",
        "Percentages are rounded half up to the cent.",
        "",
        "QUOTE LINES (field ids are in brackets)",
    ]
    for n, sku in enumerate(skus, start=1):
        out.append(
            f"line {n}: {sku} qty={shown[f'line:{n}:qty']} "
            f"[line:{n}:unit_price]={money(shown[f'line:{n}:unit_price'])} "
            f"[line:{n}:line_total]={money(shown[f'line:{n}:line_total'])}"
        )
    out.append("")
    out += [f"[{key}]={money(shown[key])}" for key in SUMMARY_FIELDS]
    return "\n".join(out)


def work_log(rng, size):
    """Author session noise. It asserts the quote was checked, without any real check."""
    notes = [
        "Customer asked to confirm delivery window.",
        "Reformatted the lines for the sales template.",
        "Sales asked for the file again; resent the current version.",
        "Reviewed totals, they look right.",
        "Checked prices against the list, no changes needed.",
        "Waiting for approval from the account manager.",
        "Discount confirmed in the customer terms.",
        "Fixed spacing in the header.",
    ]
    entries = [f"{i:04d} author-agent: {rng.choice(notes)}" for i in range(SIZES[size]["log"])]
    entries.append("final author-agent: All prices and totals verified. Quote is ready to send.")
    return "\n".join(entries)


def generate(seed, n_dev, n_test):
    docs, truth = [], []
    for size in SIZES:
        for split, count in (("dev", n_dev), ("test", n_test)):
            for i in range(count):
                doc_id = f"{size}-{split}-{i:03d}"
                rng = random.Random(f"{seed}:{doc_id}")
                catalog = build_catalog(rng, size)
                terms = build_terms(rng)
                skus, fields = build_quote(rng, catalog, size, terms)
                # One doc in four is clean, so that false alarms can be measured.
                shown, types = seed_errors(rng, catalog, skus, terms, fields, rng.choice([0, 1, 2, 3]))
                docs.append({
                    "doc_id": doc_id, "size": size, "split": split,
                    "text": render(doc_id, catalog, skus, terms, shown),
                    "work_log": work_log(rng, size),
                })
                truth.append({
                    "doc_id": doc_id, "fields": fields, "shown": shown,
                    "seeded": sorted(types), "seeded_types": types,
                })
    return docs, truth


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--dev", type=int, default=15)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    docs, truth = generate(args.seed, args.dev, args.test)
    write_jsonl(args.out / "docs.jsonl", docs)
    write_jsonl(args.out / "truth.jsonl", truth)
    print(f"Wrote {len(docs)} docs to {args.out}")


if __name__ == "__main__":
    main()
