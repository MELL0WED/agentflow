import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agentflow.metrics import parse_number

SYSTEM_PROMPT = (
    "You answer questions about a company's financial report using only the provided context. "
    "Respond with only the final number. Do not include words, units or explanation."
)


class AgentState(TypedDict, total=False):
    doc_id: str
    question: str
    chunks: list[dict]
    raw_answer: str
    answer: float | None
    llm_calls: int
    prompt_tokens: int
    timings: dict[str, float]


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))


def build_graph(store, embedder, llm, top_k: int):
    def retrieve(state: AgentState) -> AgentState:
        start = time.perf_counter()
        vector = embedder.embed([state["question"]])[0]
        chunks = store.search(state["doc_id"], vector, top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return {"chunks": chunks, "timings": {**state.get("timings", {}), "retrieve_ms": elapsed}}

    def generate(state: AgentState) -> AgentState:
        prompt = f"Context:\n{format_context(state['chunks'])}\n\nQuestion: {state['question']}\nAnswer:"
        result = llm.complete(SYSTEM_PROMPT, prompt)
        return {
            "raw_answer": result.text,
            "answer": parse_number(result.text),
            "llm_calls": state.get("llm_calls", 0) + 1,
            "prompt_tokens": state.get("prompt_tokens", 0) + result.prompt_tokens,
            "timings": {**state.get("timings", {}), "generate_ms": result.latency_ms},
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()