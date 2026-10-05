import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: float
    qdrant_url: str | None
    qdrant_path: str
    collection: str
    embed_model: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    data_path: str

    def public(self) -> dict:
        data = dict(self.__dict__)
        data.pop("llm_api_key")
        return data


def get_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        llm_model=os.getenv("LLM_MODEL", "meta-llama-3-8b-instruct"),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_path=os.getenv("QDRANT_PATH", "./qdrant_data"),
        collection=os.getenv("COLLECTION", "finqa_fixed"),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_strategy=os.getenv("CHUNK_STRATEGY", "fixed"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "3")),
        data_path=os.getenv("DATA_PATH", "data/finqa_test.json"),
    )