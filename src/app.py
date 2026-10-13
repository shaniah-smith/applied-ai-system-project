"""VibeFinder AI command-line app.

Examples (run from the project folder):
    python -m src.app "chill music for studying late at night"
    python -m src.app "songs for the gym" --trace       # show every agent step
    python -m src.app "sad songs" --offline             # no Gemini, keyword parser + template answer
    python -m src.app "happy songs" --no-rag           # baseline: Gemini without retrieval
    python -m src.app                                  # interactive mode
"""

import argparse
import json
import logging
from pathlib import Path

from src.assistant import Result, VibeFinder
from src.knowledge import load_notes
from src.llm import get_llm
from src.recommender import load_songs

LOG_DIR = Path("logs")


def setup_logging() -> None:
    """Write detailed logs to logs/vibefinder.log and keep the terminal clean."""
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "vibefinder.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_assistant(offline: bool = False) -> VibeFinder:
    """Load the catalog and knowledge notes, and connect to Gemini unless offline."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):  # hide "Loaded songs" from the recommender
        songs = load_songs("data/songs.csv")
    notes = load_notes("knowledge")
    llm = None if offline else get_llm()
    return VibeFinder(songs, notes, llm=llm)


def print_result(result: Result, show_trace: bool = False) -> None:
    """Print one answer with its retrieved sources, confidence, and guardrail status."""
    print(f"> {result.query}\n")
    print(result.answer)
    print()
    if result.songs:
        print("Retrieved songs: " + ", ".join(f"{s['title']} ({s['score']:.2f})" for s in result.songs))
    if result.notes:
        print("Retrieved notes: " + ", ".join(f"{n['title']} [{n['source']}]" for n in result.notes))
    status = "passed" if result.check_passed else "FAILED: " + "; ".join(result.problems)
    print(f"Answer source: {result.source} | Confidence: {result.confidence:.2f} ({result.confidence_label}) "
          f"| Guardrail check: {status}")
    if show_trace:
        print("\nAgent trace:")
        for step in result.trace:
            print(f"  [{step['step']}] {json.dumps(step['detail'], default=str)}")
    print()


def save_trace(result: Result) -> None:
    """Append the full trace to logs/traces.jsonl for later review."""
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": result.query, "source": result.source, "trace": result.trace},
                           default=str) + "\n")


def main() -> None:
    """Parse arguments and answer one request, or loop in interactive mode."""
    parser = argparse.ArgumentParser(description="VibeFinder AI - RAG music assistant")
    parser.add_argument("query", nargs="?", help="what you want to listen to")
    parser.add_argument("--offline", action="store_true", help="don't call Gemini")
    parser.add_argument("--no-rag", action="store_true", help="baseline: ask Gemini without retrieval")
    parser.add_argument("--trace", action="store_true", help="print every agent step")
    args = parser.parse_args()

    setup_logging()
    assistant = build_assistant(offline=args.offline)
    mode = "Gemini" if assistant.llm else "offline (no Gemini key found or --offline)"
    print(f"VibeFinder AI | mode: {mode}\n")

    if args.query is not None:
        result = assistant.run(args.query, use_rag=not args.no_rag)
        print_result(result, args.trace)
        save_trace(result)
        return

    print("Type a request (or 'quit').")
    while True:
        try:
            query = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"quit", "exit", "q"}:
            break
        result = assistant.run(query)
        print_result(result, args.trace)
        save_trace(result)


if __name__ == "__main__":
    main()
