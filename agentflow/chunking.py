from dataclasses import dataclass

from agentflow.finqa import Document


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


def chunk_document(doc: Document, strategy: str, size: int, overlap: int) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_size_chunks(doc, size, overlap)
    raise ValueError(f"unknown chunk strategy: {strategy}")