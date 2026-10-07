import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from agentflow.config import get_settings
from agentflow.evaluation import error_record, score, summarize
from agentflow.finqa import load_finqa, sample_questions
from agentflow.pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    print(f"collection {settings.collection} | model {settings.llm_model} | top_k {settings.top_k}")
    docs, questions = load_finqa(settings.data_path)
    sample = sample_questions(questions, args.n, args.seed)
    pipeline = build_pipeline(settings)

    records = []
    try:
        pipeline.llm.complete("Reply with OK.", "OK?", max_tokens=4)
        for i, q in enumerate(sample, 1):
            start = time.perf_counter()
            try:
                result = pipeline.ask(q.doc_id, q.question)
                record = score(q, docs[q.doc_id], result, (time.perf_counter() - start) * 1000)
            except Exception as exc:
                record = error_record(q, f"{type(exc).__name__}: {exc}")
            records.append(record)
            mark = "ok " if record["correct"] else "bad"
            print(f"[{i:4}/{len(sample)}] {mark} {record['category']:<20} pred={record.get('pred')} gold={q.gold:g}")
    finally:
        pipeline.close()

    summary = summarize(records)
    out = Path("results") / f"run_{args.name}.json"
    out.write_text(
        json.dumps(
            {
                "name": args.name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "seed": args.seed,
                "settings": settings.public(),
                "summary": summary,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    for key, value in summary.items():
        print(f"{key:<22} {value:.3f}" if isinstance(value, float) else f"{key:<22} {value}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()