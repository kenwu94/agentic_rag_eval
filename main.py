import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from pipeline.ingestion import ingest_documents
from pipeline.retrieval import load_index
from pipeline.agent import run_agent

load_dotenv()

DOCS_DIR = "data/sample_docs"
CHROMA_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"
TRACES_DIR = "./traces"

console = Console()


def cmd_ingest():
    console.print("[bold green]Ingesting documents...[/bold green]")
    ingest_documents(docs_dir=DOCS_DIR, persist_dir=CHROMA_DIR, bm25_path=BM25_PATH)


def cmd_query(question: str, save_trace: bool = True):
    client = OpenAI()
    collection, bm25, chunks, metadatas = load_index(persist_dir=CHROMA_DIR, bm25_path=BM25_PATH)

    console.print(f"\n[bold blue]Query:[/bold blue] {question}\n")

    trace = run_agent(
        query=question,
        collection=collection,
        bm25=bm25,
        chunks=chunks,
        metadatas=metadatas,
        openai_client=client,
    )

    console.print(f"[bold green]Answer:[/bold green] {trace.answer}")

    if trace.rewritten_queries:
        console.print(f"\n[bold yellow]Rewritten queries:[/bold yellow] {trace.rewritten_queries}")

    table = Table(title="Retrieval Steps", show_lines=True)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Strategy", style="magenta")
    table.add_column("Query", style="green", max_width=50)
    table.add_column("Docs", style="yellow", justify="right")
    table.add_column("Latency (ms)", style="red", justify="right")

    for i, step in enumerate(trace.retrieval_steps, 1):
        q = step.query if len(step.query) <= 50 else step.query[:47] + "..."
        table.add_row(str(i), step.strategy, q, str(len(step.results)), f"{step.latency_ms:.0f}")

    console.print(table)
    console.print(f"[bold]Total latency:[/bold] {trace.total_latency_ms:.0f} ms")

    if save_trace:
        Path(TRACES_DIR).mkdir(exist_ok=True)
        trace_path = f"{TRACES_DIR}/trace_{int(time.time())}.json"
        trace.save(trace_path)
        console.print(f"[dim]Trace saved → {trace_path}[/dim]")

    return trace


DEMO_QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "What were the main causes of the French Revolution?",
    "How does DNA replication work?",
    "What are the primary drivers of climate change?",
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "ingest":
            cmd_ingest()
        elif cmd == "query":
            question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEMO_QUESTIONS[0]
            cmd_query(question)
        else:
            console.print(f"Unknown command: {cmd}. Use 'ingest' or 'query <question>'.")
    else:
        # Demo: ingest then run sample queries
        cmd_ingest()
        for q in DEMO_QUESTIONS:
            cmd_query(q)
            console.print("\n" + "=" * 80 + "\n")
