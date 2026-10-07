import argparse
import json
import random
from pathlib import Path


def load_rows(name: str) -> dict[str, dict]:
    data = json.loads(Path("results", f"retrieval_{name}.json").read_text(encoding="utf-8"))
    return {r["qid"]: r for r in data["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("base_k", type=int)
    parser.add_argument("new")
    parser.add_argument("new_k", type=int)
    parser.add_argument("--boot", type=int, default=5000)
    args = parser.parse_args()

    base, new = load_rows(args.base), load_rows(args.new)
    qids = sorted(set(base) & set(new))
    n = len(qids)
    a = [base[q][f"recall@{args.base_k}"] == 1.0 for q in qids]
    b = [new[q][f"recall@{args.new_k}"] == 1.0 for q in qids]
    chars_a = sum(base[q][f"chars@{args.base_k}"] for q in qids) / n
    chars_b = sum(new[q][f"chars@{args.new_k}"] for q in qids) / n

    only_base = sum(x and not y for x, y in zip(a, b))
    only_new = sum(y and not x for x, y in zip(a, b))
    diff = (sum(b) - sum(a)) / n

    rng = random.Random(0)
    diffs = []
    for _ in range(args.boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    low, high = diffs[int(0.025 * args.boot)], diffs[int(0.975 * args.boot) - 1]

    print(f"questions            {n}")
    print(f"{args.base}@{args.base_k:<14} full {sum(a) / n:.1%}  chars {chars_a:.0f}")
    print(f"{args.new}@{args.new_k:<14} full {sum(b) / n:.1%}  chars {chars_b:.0f}")
    print(f"only base found all  {only_base}")
    print(f"only new found all   {only_new}")
    print(f"difference           {diff:+.1%}  (95% CI {low:+.1%} to {high:+.1%})")


if __name__ == "__main__":
    main()