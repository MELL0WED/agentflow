import time
from dataclasses import dataclass

from openai import OpenAI

from agentflow.config import Settings


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=2,
        )
        self.model = settings.llm_model

    def complete(self, system: str, user: str, max_tokens: int = 64) -> LLMResult:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if not response.choices:
            raise LLMError(f"no choices in response from {self.model}: {response.model_dump()}")
        usage = response.usage
        return LLMResult(
            text=(response.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )