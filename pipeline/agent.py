import json
import time
from typing import List, Dict, Any

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from pipeline.retrieval import dense_retrieve, sparse_retrieve, hybrid_retrieve
from pipeline.reranker import rerank
from pipeline.tracer import Trace


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document corpus. Use 'dense' for semantic/conceptual queries, "
                "'sparse' for exact keyword matching, and 'hybrid' for broad coverage. "
                "You may call this multiple times with different queries or strategies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Rewrite or decompose the original question if helpful.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["dense", "sparse", "hybrid"],
                        "description": "Retrieval strategy.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query", "strategy"],
            },
        },
    }
]

_SYSTEM_PROMPT = """\
You are a research assistant with access to a document retrieval tool.

Steps:
1. Analyze the question and pick the best retrieval strategy (dense = conceptual, sparse = keywords, hybrid = both).
2. Call search_documents. You may rewrite the query to improve recall.
3. Inspect results. If they don't answer the question, retry with a different strategy or query.
4. After gathering sufficient context, write a clear, grounded answer using only retrieved information.
5. If the corpus lacks the information, say so explicitly.

You may call search_documents up to 3 times. Then generate your final answer."""


def run_agent(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    chunks: List[str],
    metadatas: List[Dict],
    openai_client: OpenAI,
    model: str = "gpt-4o-mini",
    top_k: int = 5,
    use_reranker: bool = True,
    max_tool_calls: int = 3,
) -> Trace:
    """
    Agentic RAG loop: plan → retrieve (with optional query rewriting) → reflect → generate.
    Returns a Trace with all steps logged.
    """
    trace = Trace(query=query)
    t_start = time.time()

    messages: List[Any] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    tool_call_count = 0
    all_context: List[Dict] = []

    while True:
        call_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if tool_call_count < max_tool_calls:
            call_kwargs["tools"] = TOOLS
            call_kwargs["tool_choice"] = "auto"

        response = openai_client.chat.completions.create(**call_kwargs)
        msg = response.choices[0].message

        trace.agent_steps.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            }
        )
        messages.append(msg)

        if not msg.tool_calls:
            trace.answer = msg.content or ""
            break

        for tc in msg.tool_calls:
            tool_call_count += 1
            args = json.loads(tc.function.arguments)
            search_query = args["query"]
            strategy = args.get("strategy", "hybrid")
            k = int(args.get("top_k", top_k))

            if search_query != query and search_query not in trace.rewritten_queries:
                trace.rewritten_queries.append(search_query)

            t0 = time.time()
            if strategy == "dense":
                results = dense_retrieve(search_query, collection, openai_client, top_k=k)
            elif strategy == "sparse":
                results = sparse_retrieve(search_query, bm25, chunks, metadatas, top_k=k)
            else:
                results = hybrid_retrieve(
                    search_query, collection, bm25, chunks, metadatas, openai_client, top_k=k
                )
            latency = (time.time() - t0) * 1000

            if use_reranker and results:
                results = rerank(search_query, results, top_k=k)

            trace.add_retrieval_step(strategy, search_query, k, results, latency)
            all_context.extend(results)

            context_text = "\n\n---\n\n".join(
                f"[Source: {r['metadata']['source']}, chunk {r['metadata']['chunk_index']}]\n{r['text']}"
                for r in results
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Retrieved {len(results)} documents ({strategy}):\n\n{context_text}",
                }
            )

    seen: set = set()
    for r in all_context:
        if r["text"] not in seen:
            seen.add(r["text"])
            trace.final_context.append(r)

    trace.total_latency_ms = (time.time() - t_start) * 1000
    return trace
