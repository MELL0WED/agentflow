import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from agentflow.config import get_settings
from agentflow.finqa import load_finqa, sample_questions
from agentflow.metrics import evidence_recall, percentile
from agentflow.store import Embedder, VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[3])
    args = parser.parse_args()

    settings = get_settings()
    print(f"collection {settings.collection} | strategy {settings.chunk_strategy}")
    docs, questions = load_finqa(settings.data_path)
    sample = sample_questions(questions, args.n, args.seed)
    embedder = Embedder(settings.embed_model)
    store = VectorStore(settings, embedder.dim)
    max_k = max(args.k)

    rows, latencies = [], []
    try:
        for q in sample:
            start = time.perf_counter()
            vector = embedder.embed([q.question])[0]
            hits = store.search(q.doc_id, vector, max_k)
            latencies.append((time.perf_counter() - start) * 1000)
            doc = docs[q.doc_id]
            row = {"qid": q.qid, "retrieved": [h["index"] for h in hits]}
            for k in args.k:
                top = hits[:k]
                row[f"recall@{k}"] = evidence_recall(doc, q.gold_units, [(h["start"], h["end"]) for h in top])
                row[f"chars@{k}"] = sum(len(h["text"]) for h in top)
            rows.append(row)
    finally:
        store.close()

    summary = {"n": len(rows)}
    for k in args.k:
        values = [r[f"recall@{k}"] for r in rows if r[f"recall@{k}"] is not None]
        summary[f"recall@{k}"] = sum(values) / len(values)
        summary[f"full@{k}"] = sum(v == 1.0 for v in values) / len(values)
        summary[f"chars@{k}"] = sum(r[f"chars@{k}"] for r in rows) / len(rows)
    summary["retrieve_p50_ms"] = percentile(latencies, 50)
    summary["retrieve_p95_ms"] = percentile(latencies, 95)

    out = Path("results") / f"retrieval_{args.name}.json"
    out.write_text(
        json.dumps(
            {
                "name": args.name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "seed": args.seed,
                "settings": settings.public(),
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for key, value in summary.items():
        print(f"{key:<18} {value:.3f}" if isinstance(value, float) else f"{key:<18} {value}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()