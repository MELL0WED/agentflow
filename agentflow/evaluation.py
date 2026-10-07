from collections import Counter

from agentflow.finqa import Document, Question
from agentflow.metrics import evidence_recall, is_bare_number, is_correct, percentile


def score(question: Question, doc: Document, result: dict, total_ms: float) -> dict:
    chunks = result.get("chunks", [])
    raw = result.get("raw_answer", "")
    pred = result.get("answer")
    correct = is_correct(pred, question.gold, question.gold_text)
    recall = evidence_recall(doc, question.gold_units, [(c["start"], c["end"]) for c in chunks])

    if correct:
        category = "correct"
    elif pred is None:
        category = "no_number"
    elif recall is not None and recall < 1.0:
        category = "missing_evidence"
    else:
        category = "wrong_with_evidence"

    return {
        "qid": question.qid,
        "doc_id": question.doc_id,
        "question": question.question,
        "gold": question.gold,
        "gold_text": question.gold_text,
        "program": question.program,
        "pred": pred,
        "raw_answer": raw,
        "correct": correct,
        "bare_number": is_bare_number(raw),
        "evidence_recall": recall,
        "category": category,
        "retrieved": [c.get("index") for c in chunks],
        "llm_calls": result.get("llm_calls", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "timings": result.get("timings", {}),
        "total_ms": total_ms,
    }


def error_record(question: Question, message: str) -> dict:
    return {
        "qid": question.qid,
        "doc_id": question.doc_id,
        "question": question.question,
        "gold": question.gold,
        "correct": False,
        "category": "error",
        "error": message,
    }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    ok = [r for r in records if r["category"] != "error"]
    recalls = [r["evidence_recall"] for r in ok if r.get("evidence_recall") is not None]
    totals = [r["total_ms"] for r in ok]
    generate = [r["timings"].get("generate_ms", 0.0) for r in ok]
    retrieve = [r["timings"].get("retrieve_ms", 0.0) for r in ok]
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
        "evidence_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "full_evidence_rate": sum(1 for x in recalls if x == 1.0) / len(recalls) if recalls else 0.0,
        "bare_number_rate": sum(r["bare_number"] for r in ok) / len(ok) if ok else 0.0,
        "categories": dict(Counter(r["category"] for r in records)),
        "latency_p50_ms": percentile(totals, 50),
        "latency_p95_ms": percentile(totals, 95),
        "retrieve_p50_ms": percentile(retrieve, 50),
        "generate_p50_ms": percentile(generate, 50),
        "llm_calls_per_q": sum(r["llm_calls"] for r in ok) / len(ok) if ok else 0.0,
        "prompt_tokens_per_q": sum(r["prompt_tokens"] for r in ok) / len(ok) if ok else 0.0,
    }