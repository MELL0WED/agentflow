from dataclasses import dataclass

from agentflow.config import Settings
from agentflow.graph import build_graph
from agentflow.llm import LLMClient
from agentflow.store import Embedder, VectorStore


@dataclass
class Pipeline:
    graph: object
    store: VectorStore
    embedder: Embedder
    llm: LLMClient

    def ask(self, doc_id: str, question: str) -> dict:
        return self.graph.invoke({"doc_id": doc_id, "question": question})

    def close(self) -> None:
        self.store.close()


def build_pipeline(settings: Settings) -> Pipeline:
    embedder = Embedder(settings.embed_model)
    store = VectorStore(settings, embedder.dim)
    llm = LLMClient(settings)
    graph = build_graph(store, embedder, llm, settings.top_k)
    return Pipeline(graph, store, embedder, llm)