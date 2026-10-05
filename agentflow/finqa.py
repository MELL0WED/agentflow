import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Unit:
    unit_id: str
    kind: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    units: tuple[Unit, ...]
    table: tuple[tuple[str, ...], ...]

    def unit(self, unit_id: str) -> Unit:
        for u in self.units:
            if u.unit_id == unit_id:
                return u
        raise KeyError(unit_id)


@dataclass(frozen=True)
class Question:
    qid: str
    doc_id: str
    question: str
    gold: float
    gold_text: str
    program: str
    gold_units: tuple[str, ...]


def _clean(text: str) -> str:
    return " ".join(text.split())


def render_document(doc_id: str, pre_text: list[str], table: list[list[str]], post_text: list[str]) -> Document:
    pieces = [(f"text_{i}", "text", s) for i, s in enumerate(pre_text)]
    pieces += [(f"table_{i}", "table", " | ".join(_clean(c) for c in row)) for i, row in enumerate(table)]
    pieces += [(f"text_{len(pre_text) + i}", "text", s) for i, s in enumerate(post_text)]

    units, lines, cursor = [], [], 0
    for unit_id, kind, raw in pieces:
        text = _clean(raw)
        units.append(Unit(unit_id, kind, text, cursor, cursor + len(text)))
        lines.append(text)
        cursor += len(text) + 1

    clean_table = tuple(tuple(_clean(c) for c in row) for row in table)
    return Document(doc_id, "\n".join(lines), tuple(units), clean_table)


def load_finqa(path: str | Path) -> tuple[dict[str, Document], list[Question]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: dict[str, Document] = {}
    questions: list[Question] = []
    for ex in raw:
        doc_id = ex["filename"]
        if doc_id not in docs:
            docs[doc_id] = render_document(doc_id, ex["pre_text"], ex["table"], ex["post_text"])
        qa = ex["qa"]
        if isinstance(qa["exe_ans"], bool) or not isinstance(qa["exe_ans"], (int, float)):
            continue
        questions.append(
            Question(
                qid=ex["id"],
                doc_id=doc_id,
                question=qa["question"],
                gold=float(qa["exe_ans"]),
                gold_text=str(qa["answer"]),
                program=qa["program"],
                gold_units=tuple(qa["gold_inds"]),
            )
        )
    return docs, questions


def sample_questions(questions: list[Question], n: int, seed: int = 42) -> list[Question]:
    ordered = sorted(questions, key=lambda q: q.qid)
    return random.Random(seed).sample(ordered, min(n, len(ordered)))