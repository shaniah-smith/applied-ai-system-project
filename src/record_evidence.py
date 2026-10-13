"""Run the demo requests and save real outputs as markdown evidence (and paste them into README.md).

    python -m src.record_evidence

Writes:
    assets/sample_interactions.md   3 example runs with full agent traces
    assets/rag_comparison.md        same request with and without retrieval
    assets/evaluation_results.md    test harness table
"""

import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

from src.app import build_assistant, print_result, setup_logging

EXAMPLES = [
    "chill music for studying late at night",
    "I need songs to get hyped at the gym",
    "k-pop dance hits for a party",
]
COMPARE_QUERY = "cozy songs for a rainy afternoon"


def capture(fn, *args, **kwargs) -> str:
    """Return whatever fn prints."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args, **kwargs)
    return buffer.getvalue().rstrip()


def fill(readme: str, marker: str, text: str) -> str:
    """Replace the fenced block that contains the marker (or the previous output) with new text."""
    pattern = rf"(<!-- {marker} -->\n```text\n).*?(\n```)"
    return re.sub(pattern, lambda m: m.group(1) + text + m.group(2), readme, flags=re.S)


def main() -> None:
    """Record every piece of evidence and update README.md."""
    setup_logging()
    assistant = build_assistant()
    mode = "Gemini" if assistant.llm else "offline (no GEMINI_API_KEY found)"
    print(f"Recording evidence in mode: {mode}")
    if not assistant.llm:
        print("⚠️  WARNING: Gemini is not connected, so this evidence will only show offline mode.")
        print("   Put your key in .env (see README step 3), then run this again.")

    examples = []
    for query in EXAMPLES:
        result = assistant.run(query)
        examples.append(capture(print_result, result, show_trace=True))

    Path("assets").mkdir(exist_ok=True)
    Path("assets/sample_interactions.md").write_text(
        f"# Sample Interactions\n\nRecorded with `python -m src.record_evidence` (mode: {mode}).\n"
        "Each run shows the answer, the retrieved sources, the confidence score, the guardrail result, "
        "and the full agent trace (plan -> retrieve -> generate -> check -> retry/fallback).\n\n"
        + "\n\n".join(f"## Example {i}\n\n```text\n{text}\n```" for i, text in enumerate(examples, 1)) + "\n",
        encoding="utf-8")

    with_rag = assistant.run(COMPARE_QUERY)
    without_rag = assistant.run(COMPARE_QUERY, use_rag=False)
    if without_rag.source == "error":
        without_rag.answer = without_rag.answer[:160] + ("..." if len(without_rag.answer) > 160 else "")
    baseline_check = ("skipped (needs Gemini)" if without_rag.source == "error" else
                      "passed" if without_rag.check_passed else "FAILED: " + "; ".join(without_rag.problems))
    compare = (f"WITHOUT retrieval (Gemini alone):\n{without_rag.answer}\n"
               f"Guardrail check: {baseline_check}"
               f"\n\nWITH retrieval (VibeFinder RAG):\n{with_rag.answer}\n"
               f"Guardrail check: {'passed' if with_rag.check_passed else 'FAILED'} | "
               f"Retrieved notes: {', '.join(n['title'] for n in with_rag.notes) or 'none'}")
    Path("assets/rag_comparison.md").write_text(
        f"# RAG Before / After\n\nRequest: `{COMPARE_QUERY}` (mode: {mode})\n\n```text\n{compare}\n```\n",
        encoding="utf-8")

    # Reuse a saved evaluation so we don't send 9 more requests (Gemini's free tier rate-limits).
    # Pass --rerun-eval to run the harness again.
    saved = Path("assets/evaluation_results.md")
    if saved.exists() and "--rerun-eval" not in sys.argv:
        text = saved.read_text(encoding="utf-8")
        evaluation = text[text.index("Mode:"):text.index("**", text.index("Mode:"))].rstrip()
        evaluation += "\n" + re.search(r"\*\*\d+ out of .*", text).group(0)
        print("Using saved assets/evaluation_results.md (pass --rerun-eval to run it again)")
    else:
        evaluation = subprocess.run([sys.executable, "-m", "src.evaluate", "--save"],
                                    capture_output=True, text=True).stdout
        evaluation = evaluation.split("\nSaved to")[0].rstrip()

    readme = Path("README.md").read_text(encoding="utf-8")
    for i, text in enumerate(examples, 1):
        readme = fill(readme, f"EXAMPLE {i}", text)
    readme = fill(readme, "RAG COMPARISON", compare)
    readme = fill(readme, "EVALUATION", evaluation)
    Path("README.md").write_text(readme, encoding="utf-8")
    print("✅ Saved assets/sample_interactions.md and assets/rag_comparison.md")
    print("✅ Pasted the real outputs into README.md")


if __name__ == "__main__":
    main()
