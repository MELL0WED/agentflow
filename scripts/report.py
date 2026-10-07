import json
from pathlib import Path

COLUMNS = [
    ("accuracy", "acc", "{:.1%}"),
    ("evidence_recall", "ev_recall", "{:.1%}"),
    ("full_evidence_rate", "full_ev", "{:.1%}"),
    ("bare_number_rate", "bare_num", "{:.1%}"),
    ("latency_p50_ms", "p50_ms", "{:.0f}"),
    ("latency_p95_ms", "p95_ms", "{:.0f}"),
    ("llm_calls_per_q", "llm/q", "{:.2f}"),
    ("prompt_tokens_per_q", "tok/q", "{:.0f}"),
]


def main() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in Path("results").glob("run_*.json")]
    runs.sort(key=lambda r: r["created"])
    header = ["run", "n"] + [c[1] for c in COLUMNS] + ["categories"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for run in runs:
        s = run["summary"]
        cells = [run["name"], str(s["n"])] + [fmt.format(s[key]) for key, _, fmt in COLUMNS]
        cells.append(", ".join(f"{k}={v}" for k, v in sorted(s["categories"].items())))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()