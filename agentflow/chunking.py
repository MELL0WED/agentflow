from dataclasses import dataclass

from agentflow.finqa import Document, Unit


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int
    end: int

    def payload(self) -> dict:
        return {"doc_id": self.doc_id, "index": self.index, "text": self.text, "start": self.start, "end": self.end}


def fixed_size_chunks(doc: Document, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        chunks.append(Chunk(doc.doc_id, len(chunks), doc.text[start:end], start, end))
        if end == len(doc.text):
            break
        start += step
    return chunks


def _group(units: list[Unit], size: int) -> list[list[Unit]]:
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    length = 0
    for u in units:
        if current and length + 1 + len(u.text) > size:
            groups.append(current)
            current, length = [], 0
        length += len(u.text) + (1 if current else 0)
        current.append(u)
    if current:
        groups.append(current)
    return groups


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    space = cut.find(" ")
    return cut[space + 1 :] if space != -1 else cut


def structured_chunks(doc: Document, size: int, caption_chars: int = 200) -> list[Chunk]:
    table = [u for u in doc.units if u.kind == "table"]
    text = [u for u in doc.units if u.kind == "text"]
    table_start = table[0].start if table else None
    pre = [u for u in text if table_start is None or u.end < table_start]
    post = [u for u in text if table_start is not None and u.start > table_start]

    chunks: list[Chunk] = []

    def add(body: str, start: int, end: int) -> None:
        chunks.append(Chunk(doc.doc_id, len(chunks), body, start, end))

    for g in _group(pre, size):
        add("\n".join(u.text for u in g), g[0].start, g[-1].end)

    if table:
        header, rows = table[0], table[1:]
        caption = _tail(" ".join(u.text for u in pre), caption_chars)
        prefix = "\n".join(p for p in (caption, header.text) if p)
        if not rows:
            add(prefix, header.start, header.end)
        budget = max(size - len(prefix) - 1, 100)
        for i, g in enumerate(_group(rows, budget)):
            start = header.start if i == 0 else g[0].start
            add(prefix + "\n" + "\n".join(u.text for u in g), start, g[-1].end)

    for g in _group(post, size):
        add("\n".join(u.text for u in g), g[0].start, g[-1].end)
    return chunks


def chunk_document(doc: Document, strategy: str, size: int, overlap: int) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_size_chunks(doc, size, overlap)
    if strategy == "structured":
        return structured_chunks(doc, size)
    raise ValueError(f"unknown chunk strategy: {strategy}")