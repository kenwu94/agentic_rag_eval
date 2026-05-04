from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json


@dataclass
class RetrievalStep:
    strategy: str
    query: str
    top_k: int
    results: List[Dict[str, Any]]
    latency_ms: float


@dataclass
class Trace:
    query: str
    rewritten_queries: List[str] = field(default_factory=list)
    retrieval_steps: List[RetrievalStep] = field(default_factory=list)
    final_context: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    total_latency_ms: float = 0.0
    agent_steps: List[Dict[str, Any]] = field(default_factory=list)

    def add_retrieval_step(
        self, strategy: str, query: str, top_k: int, results: List[Dict], latency_ms: float
    ):
        self.retrieval_steps.append(
            RetrievalStep(strategy=strategy, query=query, top_k=top_k, results=results, latency_ms=latency_ms)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
