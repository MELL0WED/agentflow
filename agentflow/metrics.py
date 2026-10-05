import re
from collections.abc import Iterable

from agentflow.finqa import Document

NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+"
NUMBER_RE = re.compile(NUMBER)
BARE_RE = re.compile(rf"\s*\$?\s*({NUMBER})\s*%?\s*\.?\s*")


def _normalize(text: str) -> str:
    return text.replace("\u2212", "-").replace("\u2013", "-")


def parse_number(text: str | None) -> float | None:
    if not text:
        return None
    text = _normalize(text)
    bare = BARE_RE.fullmatch(text)
    if bare:
        return float(bare.group(1).replace(",", ""))
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def is_bare_number(text: str | None) -> bool:
    return bool(text) and BARE_RE.fullmatch(_normalize(text)) is not None


def is_correct(pred: float | None, gold: float, gold_text: str = "", rel_tol: float = 0.01) -> bool:
    if pred is None:
        return False
    candidates = [gold, gold * 100]
    shown = parse_number(gold_text)
    if shown is not None:
        candidates.append(shown)
    for c in candidates:
        if c == 0:
            if abs(pred) < 1e-9:
                return True
        elif abs(pred - c) <= rel_tol * abs(c):
            return True
    return False


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def evidence_recall(doc: Document, gold_units: Iterable[str], spans: Iterable[tuple[int, int]]) -> float | None:
    gold = list(gold_units)
    if not gold:
        return None
    covered = merge_ranges(spans)
    hits = 0
    for unit_id in gold:
        u = doc.unit(unit_id)
        if any(s <= u.start and u.end <= e for s, e in covered):
            hits += 1
    return hits / len(gold)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)