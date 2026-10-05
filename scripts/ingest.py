import time

from agentflow.chunking import chunk_document
from agentflow.config import get_settings
from agentflow.finqa import load_finqa
from agentflow.store import Embedder, VectorStore


def main() -> None:
    settings = get_settings()
    docs, _ = load_finqa(settings.data_path)

    chunks = []
    for doc in docs.values():
        chunks.extend(chunk_document(doc, settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap))

    embedder = Embedder(settings.embed_model)
    start = time.perf_counter()
    vectors = embedder.embed([c.text for c in chunks])
    embed_s = time.perf_counter() - start

    store = VectorStore(settings, embedder.dim)
    try:
        store.recreate()
        start = time.perf_counter()
        store.upsert(chunks, vectors)
        upsert_s = time.perf_counter() - start
    finally:
        store.close()

    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    print(f"collection     {settings.collection}")
    print(f"strategy       {settings.chunk_strategy}")
    print(f"documents      {len(docs)}")
    print(f"chunks         {len(chunks)} ({len(chunks) / len(docs):.1f} per doc, avg {avg_len:.0f} chars)")
    print(f"embed time     {embed_s:.1f}s")
    print(f"upsert time    {upsert_s:.1f}s")


if __name__ == "__main__":
    main()